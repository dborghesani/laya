"""Fine-tune and evaluate Laya on a local typed-decisions dataset.

Example:
	python scripts/finetune.py \
		--dataset dataset/laya_automotive_typed_decisions_50k \
		--output models/laya-automotive

The dataset must expose ``state``, ``questions`` and ``gold`` columns.  A
DatasetDict with ``train``/``test`` splits, a saved Hugging Face dataset, or a
directory containing train/test Parquet files is supported.
"""
import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence, Sized
from datetime import datetime, timezone
from typing import (
	Any,
	TypedDict,
	cast,
)

import structlog
from huggingface_hub import snapshot_download
from torch.utils.tensorboard import SummaryWriter

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from datasets import Dataset, load_dataset, load_from_disk

import laya
from laya.agent import _fix_tokenizer_config
from laya.common import (
	QTYPES,
	build_model,
	build_sequence,
	proper_reward,
	render_options,
)

logger = structlog.get_logger()

JSONValue = bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"] | None
State = str | dict[str, JSONValue] | list[JSONValue]


class Question(TypedDict, total=False):
	type: str
	instructions: str | JSONValue
	criteria: dict[str, JSONValue] | list[JSONValue]


class InternalQuestion(TypedDict):
	t: str
	ins: str
	crit: Any


class TrainingItem(TypedDict):
	ids: list[int]
	markers: list[int]
	qtype: int
	target: list[float]
	label: int


class Batch(TypedDict):
	input_ids: torch.Tensor
	attention_mask: torch.Tensor
	marker_pos: torch.Tensor
	marker_mask: torch.Tensor
	target: torch.Tensor
	qtype: torch.Tensor


CalibrationRecord = tuple[int, Sequence[float], Sequence[float]]


def gpu_metrics(device: torch.device) -> dict[str, float]:
	if device.type != "cuda":
		return {}
	metrics = {
		"memory_allocated_gb": torch.cuda.memory_allocated(device) / 1024 ** 3,
		"memory_reserved_gb": torch.cuda.memory_reserved(device) / 1024 ** 3,
		"max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024 ** 3,
	}
	device_index = device.index or 0
	visible_devices = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
					   if value.strip()]
	gpu_id = visible_devices[device_index] if device_index < len(visible_devices) else str(device_index)
	try:
		result = subprocess.run(
			["nvidia-smi", f"--id={gpu_id}",
			 "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
			 "--format=csv,noheader,nounits"],
			check=True, capture_output=True, text=True,
		)
		utilization, memory_utilization, memory_used, power, temperature = (
			float(value.strip()) for value in result.stdout.splitlines()[0].split(","))
		metrics.update({
			"utilization_percent": utilization,
			"memory_utilization_percent": memory_utilization,
			"memory_used_mb": memory_used,
			"power_watts": power,
			"temperature_celsius": temperature,
		})
	except (FileNotFoundError, IndexError, subprocess.CalledProcessError, ValueError):
		pass
	return metrics


def format_duration(seconds: float) -> str:
	seconds = max(0, int(seconds))
	hours, remainder = divmod(seconds, 3600)
	minutes, seconds = divmod(remainder, 60)
	return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def temperature_closure(
	optimizer: torch.optim.Optimizer,
	targets: torch.Tensor,
	logits: torch.Tensor,
	log_temp: torch.Tensor,
):
	def closure():
		optimizer.zero_grad()
		loss = -(targets * torch.log_softmax(logits / log_temp.exp(), -1)).sum(-1).mean()
		loss.backward()
		return loss

	return closure


def parse_json(value: str | JSONValue) -> Any:
	return json.loads(value) if isinstance(value, str) else value


def load_split(path: str, split: str, limit: int | None = None) -> Dataset:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Dataset path does not exist: {path}")
	jsonl = os.path.join(path, f"{split}.jsonl")
	if os.path.exists(jsonl):
		if limit is not None:
			with open(jsonl) as handle:
				return Dataset.from_list([json.loads(line) for _, line in zip(range(limit), handle)])
		return load_dataset("json", data_files={split: jsonl}, split=split)
	try:
		loaded = load_from_disk(path)
		if hasattr(loaded, "keys"):
			loaded = loaded[split]
		return loaded.select(range(min(limit, len(loaded)))) if limit is not None else loaded
	except (FileNotFoundError, ValueError, KeyError):
		pass

	files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
	if files:
		split_files = [f for f in files if split in os.path.basename(f).lower()]
		loaded = load_dataset("parquet", data_files={split: split_files or files}, split=split)
		if split_files or "split" not in loaded.column_names:
			return loaded.select(range(min(limit, len(loaded)))) if limit is not None else loaded
		loaded = loaded.filter(lambda row: row["split"] == split)
		return loaded.select(range(min(limit, len(loaded)))) if limit is not None else loaded
	loaded = load_dataset(path, split=split)
	return loaded.select(range(min(limit, len(loaded)))) if limit is not None else loaded


def internal_question(question: Mapping[str, Any]) -> InternalQuestion:
	qtype = question["type"]
	criteria = question.get("criteria")
	if qtype == "choice" and isinstance(criteria, list):
		criteria = {key: None for key in criteria}
	instructions = question["instructions"]
	if not isinstance(instructions, str):
		instructions = json.dumps(instructions, ensure_ascii=False)
	return {"t": qtype, "ins": instructions, "crit": criteria}


def training_item(
	tokenizer: Any,
	cfg: Mapping[str, Any],
	state: State,
	question: Mapping[str, Any],
	gold: Mapping[str, Any],
) -> TrainingItem | None:
	q = internal_question(question)
	probabilities = gold.get("probabilities", {})
	if q["t"] == "choice":
		target = [float(probabilities.get(key, 0.0)) for key in q["crit"]]
	elif q["t"] == "noul":
		target = [float(probabilities.get("false", 0.5)), float(probabilities.get("true", 0.5))]
	else:
		target = [float(probabilities.get(str(i), 0.0)) for i in range(len(q["crit"]))]
	total = sum(target)
	target = [v / total for v in target] if total > 0 else [1.0 / len(target)] * len(target)
	sequence, markers = build_sequence(tokenizer, state, cast(dict[str, Any], q),
										 cfg["max_len"], cfg["head_max_len"])
	if len(markers) != len(render_options(cast(dict[str, Any], q))):
		return None
	return {"ids": sequence, "markers": markers, "qtype": QTYPES[q["t"]],
			"target": target, "label": int(np.argmax(target))}


def preprocess(dataset: Iterable[Mapping[str, Any]], tokenizer: Any,
			   cfg: Mapping[str, Any]) -> list[TrainingItem]:
	items: list[TrainingItem] = []
	for row in dataset:
		state = parse_json(row["state"])
		questions = parse_json(row["questions"])
		gold = parse_json(row["gold"])
		for qid, question in questions.items():
			if qid in gold:
				item = training_item(tokenizer, cfg, state, question, gold[qid])
				if item is not None:
					items.append(item)
	return items


def collate(batch: Sequence[TrainingItem], pad_id: int) -> Batch:
	n, length = len(batch), max(len(item["ids"]) for item in batch)
	kmax = max(len(item["markers"]) for item in batch)
	ids = torch.full((n, length), pad_id, dtype=torch.long)
	attention = torch.zeros((n, length), dtype=torch.long)
	markers = torch.zeros((n, kmax), dtype=torch.long)
	mask = torch.zeros((n, kmax), dtype=torch.bool)
	target = torch.zeros((n, kmax), dtype=torch.float32)
	for i, item in enumerate(batch):
		ids[i, :len(item["ids"])] = torch.tensor(item["ids"])
		attention[i, :len(item["ids"])] = 1
		k = len(item["markers"])
		markers[i, :k] = torch.tensor(item["markers"])
		mask[i, :k] = True
		target[i, :len(item["target"])] = torch.tensor(item["target"])
	return {"input_ids": ids, "attention_mask": attention, "marker_pos": markers,
			"marker_mask": mask, "target": target,
			"qtype": torch.tensor([item["qtype"] for item in batch])}


def fit_temperature(predictions: Sequence[CalibrationRecord]) -> list[float]:
	fitted = [1.2, 1.2, 1.2]
	for qtype in range(3):
		selected = [(z, target) for qt, z, target in predictions if qt == qtype]
		if len(selected) < 10:
			continue
		width = max(len(z) for z, _ in selected)
		logits = torch.full((len(selected), width), -1e4)
		targets = torch.zeros((len(selected), width))
		for i, (z, target) in enumerate(selected):
			logits[i, :len(z)] = torch.tensor(z)
			targets[i, :len(target)] = torch.tensor(target)
		log_temp = torch.zeros(1, requires_grad=True)
		optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=100)
		optimizer.step(temperature_closure(optimizer, targets, logits, log_temp))
		fitted[qtype] = float(torch.clamp(log_temp.exp(), 0.1, 10.0).item())
	return fitted


def train(
	items: list[TrainingItem],
	validation_items: list[TrainingItem],
	model_dir: str,
	output_dir: str,
	args: argparse.Namespace,
) -> list[float]:
	writer = None
	if args.tensorboard:
		writer = SummaryWriter(os.path.join(output_dir, "tensorboard"))
	with open(os.path.join(model_dir, "rl_agent_config.json")) as handle:
		cfg = json.load(handle)
	cfg.update({"gradient_checkpointing": True, "max_tokens_per_batch": args.max_tokens,
				"max_len": args.max_len, "head_max_len": args.head_max_len})
	tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
	model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
	model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
	device = torch.device(args.device)
	if device.type == "cuda":
		device = torch.device("cuda", 0 if device.index is None else device.index)
		torch.cuda.set_device(device)
		model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
	model.head_checkpointing = True
	model.to(device).train()
	encoder_params = [p for name, p in model.named_parameters() if "encoder." in name]
	head_params = [p for name, p in model.named_parameters() if "encoder." not in name]
	optimizer = torch.optim.AdamW([
		{"params": encoder_params, "lr": args.lr_encoder},
		{"params": head_params, "lr": args.lr_head},
	], weight_decay=0.01)
	updates = max(1, (len(items) // args.batch_size) * args.epochs // args.grad_accum)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, updates, eta_min=1e-6)
	scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
	rng = random.Random(args.seed)
	global_step = 0
	batches_per_epoch = (len(items) + args.batch_size - 1) // args.batch_size
	total_steps = args.epochs * batches_per_epoch
	training_start = time.perf_counter()
	best_checkpoints: list[tuple[float, str]] = []
	checkpoints_dir = os.path.join(output_dir, "checkpoints")

	def save_checkpoint(name: str, validation_loss: float) -> str:
		checkpoint_dir = os.path.join(checkpoints_dir, name)
		os.makedirs(checkpoint_dir, exist_ok=True)
		save_file({key: value.half().contiguous().cpu() for key, value in model.state_dict().items()},
				  os.path.join(checkpoint_dir, "model.safetensors"))
		with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as handle:
			json.dump({"epoch": epoch + 1, "validation_cross_entropy": validation_loss}, handle, indent=2)
		return checkpoint_dir

	def validation_cross_entropy() -> float:
		model.eval()
		loss_sum, item_count = 0.0, 0
		with torch.no_grad():
			for start in range(0, len(validation_items), args.eval_batch_size):
				batch_items = validation_items[start:start + args.eval_batch_size]
				batch = collate(batch_items, tokenizer.pad_token_id)
				logits, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
								  batch["marker_pos"].to(device), batch["marker_mask"].to(device),
								  batch["qtype"].to(device))
				mask = batch["marker_mask"].to(device)
				target = batch["target"].to(device)
				loss = -(target * torch.log_softmax(logits.float().masked_fill(~mask, -1e4), -1)).sum(-1)
				loss_sum += loss.sum().item()
				item_count += len(batch_items)
		model.train()
		return loss_sum / item_count

	for epoch in range(args.epochs):
		rng.shuffle(items)
		sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * epoch / max(1, args.epochs - 1)
		epoch_start = time.perf_counter()
		optimizer.zero_grad(set_to_none=True)
		for step in range(0, len(items), args.batch_size):
			batch_items = items[step:step + args.batch_size]
			batch = collate(batch_items, tokenizer.pad_token_id)
			autocast = torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda")
			with autocast:
				logits, act = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
									 batch["marker_pos"].to(device), batch["marker_mask"].to(device),
									 batch["qtype"].to(device))
			logits = logits.float()
			mask = batch["marker_mask"].to(device)
			target = batch["target"].to(device)
			k = mask.sum(-1, keepdim=True).float()
			noise = torch.randn((args.group_size,) + logits.shape, device=device) * sigma * mask
			noise = (noise - noise.sum(-1, keepdim=True) / k) * mask
			sampled = logits.detach().unsqueeze(0) + noise
			probs = torch.softmax(sampled.masked_fill(~mask, -1e4), -1)
			with torch.no_grad():
				reward = proper_reward(probs, target.unsqueeze(0), batch["qtype"].to(device), mask,
									   w_sph=0.75, w_rps=1.0)
				advantage = reward - reward.mean(0, keepdim=True)
				advantage = advantage / (advantage.std() + 1e-6)
			log_prob = -(((sampled - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
			loss_rl = -(advantage * log_prob).mean()
			loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
			loss = (loss_rl + loss_ce + 0.0 * act.sum()) / args.grad_accum
			scaler.scale(loss).backward()
			global_step += 1
			if global_step % args.log_every == 0 or step + len(batch_items) >= len(items):
				metrics = {
					"loss_total": loss.item() * args.grad_accum,
					"loss_rl": loss_rl.item(),
					"loss_cross_entropy": loss_ce.item(),
					"mean_reward": reward.mean().item(),
					"lr_encoder": optimizer.param_groups[0]["lr"],
					"lr_head": optimizer.param_groups[1]["lr"],
				}
				if writer is not None:
					writer.add_scalar("loss/total", metrics["loss_total"], global_step)
					writer.add_scalar("loss/rl", metrics["loss_rl"], global_step)
					writer.add_scalar("loss/cross_entropy", metrics["loss_cross_entropy"], global_step)
					writer.add_scalar("train/reward", metrics["mean_reward"], global_step)
					writer.add_scalar("train/lr_encoder", metrics["lr_encoder"], global_step)
					writer.add_scalar("train/lr_head", metrics["lr_head"], global_step)
					for name, value in gpu_metrics(device).items():
						writer.add_scalar(f"gpu/{name}", value, global_step)
					writer.flush()
				elapsed = time.perf_counter() - training_start
				eta = elapsed / global_step * (total_steps - global_step)
				logger.info(
					"train",
					epoch=f"{epoch + 1}/{args.epochs}",
					step=f"{step // args.batch_size + 1}/{batches_per_epoch}",
					loss=round(metrics["loss_total"], 5),
					loss_rl=round(metrics["loss_rl"], 5),
					loss_ce=round(metrics["loss_cross_entropy"], 5),
					eta=format_duration(eta),
				)
			if ((step // args.batch_size) + 1) % args.grad_accum == 0 or step + args.batch_size >= len(items):
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				old_scale = scaler.get_scale()
				scaler.step(optimizer)
				scaler.update()
				if scaler.get_scale() >= old_scale:
					scheduler.step()
				optimizer.zero_grad(set_to_none=True)
		logger.info("epoch_complete", epoch=epoch + 1, epochs=args.epochs,
					items=len(items), duration_seconds=time.perf_counter() - epoch_start)
		validation_loss = validation_cross_entropy()
		if writer is not None:
			writer.add_scalar("validation/cross_entropy", validation_loss, epoch + 1)
			writer.flush()
		logger.info("validation_complete", epoch=epoch + 1, cross_entropy=round(validation_loss, 5))
		if (epoch + 1) % args.checkpoint_every_epochs == 0:
			save_checkpoint(f"epoch-{epoch + 1}", validation_loss)
		best_checkpoint = save_checkpoint(f"best-epoch-{epoch + 1}", validation_loss)
		best_checkpoints.append((validation_loss, best_checkpoint))
		best_checkpoints.sort(key=lambda checkpoint: checkpoint[0])
		while len(best_checkpoints) > args.save_best_limit:
			_, checkpoint_dir = best_checkpoints.pop()
			shutil.rmtree(checkpoint_dir)
	if writer is not None:
		writer.close()

	model.eval()
	calibration = []
	with torch.no_grad():
		for start in range(0, min(len(items), args.calibration_items), args.eval_batch_size):
			subset = items[start:start + args.eval_batch_size]
			batch = collate(subset, tokenizer.pad_token_id)
			logits, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
							  batch["marker_pos"].to(device), batch["marker_mask"].to(device),
							  batch["qtype"].to(device))
			for row, item in zip(logits.float().cpu().numpy(), subset):
				calibration.append((item["qtype"], row[:len(item["markers"])], item["target"]))
	temperatures = fit_temperature(calibration)
	os.makedirs(output_dir, exist_ok=True)
	save_file({key: value.half().contiguous().cpu() for key, value in model.state_dict().items()},
			  os.path.join(output_dir, "model.safetensors"))
	model.encoder.config.save_pretrained(os.path.join(output_dir, "encoder"))
	tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
	cfg.update({"fine_tuned": True, "model_name": "laya-automotive-typed-decisions",
				"temperature": temperatures})
	with open(os.path.join(output_dir, "rl_agent_config.json"), "w") as handle:
		json.dump(cfg, handle, indent=2)
	logger.info(f"finetuning complete, temperatures: {temperatures}")
	return temperatures


def evaluate(
	dataset: Iterable[Mapping[str, Any]],
	output_dir: str,
	report_path: str,
	device: str,
) -> dict[str, Any]:
	agent = laya.Agent(output_dir, device=device)
	predictions, latencies = [], []
	for row in dataset:
		state, questions = parse_json(row["state"]), parse_json(row["questions"])
		gold = parse_json(row["gold"])
		start = time.perf_counter()
		answers = agent.predict(state, questions)["answers"]
		latencies.append((time.perf_counter() - start) * 1000)
		predictions.append({"workflow": row.get("workflow", "unknown"), "pred": answers,
							"gold": gold, "questions": questions})
	accuracy, soft, brier, kl, tv, conf, correct = [], [], [], [], [], [], []
	score_mae, within_one = [], []
	per_workflow: dict[str, list[float]] = {}
	for item in predictions:
		wf_correct = per_workflow.setdefault(item["workflow"], [])
		for qid, question in item["questions"].items():
			pred, gold = item["pred"][qid], item["gold"][qid]
			qtype = question["type"]
			if qtype == "choice":
				criteria = question["criteria"]
				keys = list(criteria.keys()) if isinstance(criteria, dict) else list(criteria)
				pp = np.array([pred["probabilities"].get(k, 1e-6) for k in keys], float)
				gp = np.array([gold["probabilities"].get(k, 1e-6) for k in keys], float)
			elif qtype == "noul":
				p = float(pred["noul"]); g = float(gold.get("noul", gold.get("probabilities", {}).get("true", 0.5)))
				pp, gp = np.array([1 - p, p]), np.array([1 - g, g])
			else:
				levels = len(question.get("criteria", []))
				pp = np.array([pred["probabilities"].get(str(i), 0.0) for i in range(levels)], float)
				gp = np.array([gold["probabilities"].get(str(i), 0.0) for i in range(levels)], float)
				expected = float((np.arange(len(pp)) * pp).sum()) if pp.sum() else float(pred["score"])
				actual = float(gold.get("score", gold.get("label", 0)))
				score_mae.append(abs(expected - actual)); within_one.append(float(abs(expected - actual) <= 1))
			pp /= max(pp.sum(), 1e-12); gp /= max(gp.sum(), 1e-12)
			predicted_label = int(np.argmax(pp))
			if qtype == "choice":
				gold_label = keys.index(str(gold["label"]))
			elif qtype == "noul":
				gold_label = int(str(gold["label"]).lower() == "true")
			else:
				gold_label = int(gold["label"])
			hit = float(predicted_label == gold_label)
			accuracy.append(hit); wf_correct.append(hit); conf.append(float(pp.max())); correct.append(hit)
			soft.append(float((pp * gp).sum())); brier.append(float(((pp - gp) ** 2).sum()))
			kl.append(float((gp * np.log(np.clip(gp / pp, 1e-12, 1e4))).sum()))
			tv.append(float(0.5 * np.abs(pp - gp).sum()))
	metrics = {"accuracy": float(np.mean(accuracy)), "soft_accuracy": float(np.mean(soft)),
			   "brier": float(np.mean(brier)), "ece": laya.ece_score(np.array(conf), np.array(correct)),
			   "score_mae": float(np.mean(score_mae)) if score_mae else 0.0,
			   "within_1_level": float(np.mean(within_one)) if within_one else 0.0,
			   "latency_p50_ms": float(np.percentile(latencies, 50)),
			   "latency_p95_ms": float(np.percentile(latencies, 95)),
			   "kl_divergence": float(np.mean(kl)), "total_variation": float(np.mean(tv))}
	report = {"benchmark": "laya_automotive_typed_decisions_50k", "n_cases": len(cast(Sized, dataset)),
			  "n_decisions": len(accuracy), "metrics": metrics,
			  "comparison": {"TypeSafe Jev 1.13.0": {
				  "accuracy": 0.727, "soft_accuracy": 0.580, "brier": 0.148,
				  "ece": 0.144, "score_mae": 0.391, "latency_p50_ms": 710,
				  "source": "published reference from the original typed-decisions benchmark"},
							 "Teacher Self-Agreement": {"accuracy": 0.735},
							 "Laya (fine-tuned)": metrics},
			  "per_workflow": {wf: {"n_decisions": len(values), "accuracy": float(np.mean(values))}
							   for wf, values in per_workflow.items()}}
	os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
	with open(report_path, "w") as handle:
		json.dump(report, handle, indent=2)
	logger.info(json.dumps(report, indent=2))
	return report


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", default="dataset/laya_automotive_typed_decisions_50k")
	parser.add_argument("--limit", type=int, default=None,
						help="use at most N cases from each split (useful for fast local debugging)")
	parser.add_argument("--model", default="convaiinnovations/laya")
	parser.add_argument("--output", default="output/laya_automotive_finetuned")
	parser.add_argument("--run-name", default=None,
						help="unique name for this run; defaults to a UTC timestamp and process ID")
	parser.add_argument("--report", default=None)
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--epochs", type=int, default=4)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--grad-accum", type=int, default=4)
	parser.add_argument("--group-size", type=int, default=4)
	parser.add_argument("--eval-batch-size", type=int, default=16)
	parser.add_argument("--calibration-items", type=int, default=400)
	parser.add_argument("--max-len", type=int, default=1024)
	parser.add_argument("--head-max-len", type=int, default=256)
	parser.add_argument("--max-tokens", type=int, default=4096)
	parser.add_argument("--lr-encoder", type=float, default=2.5e-5)
	parser.add_argument("--lr-head", type=float, default=1e-4)
	parser.add_argument("--sigma-start", type=float, default=0.4)
	parser.add_argument("--sigma-end", type=float, default=0.1)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True,
						help="write loss and GPU metrics to OUTPUT/runs for TensorBoard (default: enabled)")
	parser.add_argument("--log-every", type=int, default=10,
						help="record TensorBoard metrics every N batches")
	parser.add_argument("--checkpoint-every-epochs", type=int, default=1,
						help="save a regular checkpoint every N epochs")
	parser.add_argument("--save-best-limit", type=int, default=3,
						help="number of checkpoints with the lowest validation loss to retain")
	args = parser.parse_args()
	if args.limit is not None and args.limit < 1:
		parser.error("--limit must be at least 1")
	if args.log_every < 1:
		parser.error("--log-every must be at least 1")
	if args.checkpoint_every_epochs < 1:
		parser.error("--checkpoint-every-epochs must be at least 1")
	if args.save_best_limit < 1:
		parser.error("--save-best-limit must be at least 1")
	random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
	logger.info(f"preparing model {args.model}...")
	model_dir = snapshot_download(args.model)
	_fix_tokenizer_config(model_dir)
	with open(os.path.join(model_dir, "rl_agent_config.json")) as handle:
		cfg = json.load(handle)
	cfg.update({"max_len": args.max_len, "head_max_len": args.head_max_len})
	logger.info("loading tokenizer...")
	tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
	logger.info("loading dataset and preprocessing...")
	train_data = load_split(args.dataset, "train", args.limit)
	test_data = load_split(args.dataset, "test", args.limit)
	logger.info(f"loaded {len(train_data)} training cases and {len(test_data)} test cases")
	logger.info("preprocessing training data...")
	items = preprocess(train_data, tokenizer, cfg)
	validation_items = preprocess(test_data, tokenizer, cfg)
	logger.info(f"preprocessed {len(items)} training decisions from {len(train_data)} cases")
	logger.info(f"preprocessed {len(validation_items)} validation decisions from {len(test_data)} cases")
	if not validation_items:
		raise ValueError("The test split contains no valid decisions for checkpoint validation")
	run_name = args.run_name or f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{os.getpid()}"
	output_dir = os.path.join(args.output, run_name)
	if os.path.exists(output_dir):
		raise FileExistsError(f"Run directory already exists: {output_dir}")
	logger.info("starting training...")
	temperatures = train(items, validation_items, model_dir, output_dir, args)
	logger.info("calibration temperatures: %s" % [round(value, 3) for value in temperatures])
	report = args.report or os.path.join(output_dir, "benchmark_report.json")
	logger.info("starting evaluation...")
	evaluate(test_data, output_dir, report, args.device)


if __name__ == "__main__":
	main()
