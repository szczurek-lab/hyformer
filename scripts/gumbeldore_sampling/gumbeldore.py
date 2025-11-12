#!/usr/bin/env python3
"""
Hyformer + Gumbeldore sampling with Incremental SBS.
"""

import os
import sys
from typing import Callable, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from . import stochastic_beam_search as sbs
from .incremental_sbs import IncrementalSBS

from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.configs.model import ModelConfig
from hyformer.utils.tokenizers.auto import AutoTokenizer
from hyformer.models.auto import AutoModel
from hyformer.utils.properties.auto import AutoTarget
from hyformer.utils.chemistry import is_valid


class HyformerSBSAdapter:
	"""Adapter that exposes Hyformer to Incremental SBS.

	Provides three SBS-compatible callables:
	- child_log_probability_fn: batched next-token log-probabilities for states
	- child_transition_fn: appends an action (token id) to a state and flags leaves
	- leaf_evaluation_fn: scores finished sequences with an oracle (e.g., QED)
	"""
	def __init__(self, model, tokenizer, device: torch.device, *, temperature: float = 1.0, top_k: Optional[int] = None, max_new_tokens: int = 128):
		"""Initialize adapter.

		Args:
			model: Hyformer model 
			tokenizer: tokenizer exposing special ids and decode
			device: torch device
			temperature: softmax temperature for next-token sampling/logits shaping
			top_k: optional top-k truncation for next-token support
			max_new_tokens: hard cap for sequence length used to mark leaves
		"""
		self.model = model
		self.tokenizer = tokenizer
		self.device = device
		self.temperature = float(temperature)
		self.top_k = int(top_k) if top_k is not None else None
		self.max_new_tokens = int(max_new_tokens)
		self.eos_id = getattr(tokenizer, 'sep_token_id', None)
		self.pad_id = getattr(tokenizer, 'pad_token_id', None)

	def _collate_states(self, states: List[torch.Tensor]) -> torch.Tensor:
		"""Right-pad a list of 1D token id tensors to a batch on the target device."""
		lengths = [int(s.size(0)) for s in states]
		max_len = max(lengths)
		pad_id = 0 if self.pad_id is None else int(self.pad_id)
		padded = torch.full((len(states), max_len), pad_id, dtype=torch.long)
		for i, s in enumerate(states):
			padded[i, : s.size(0)] = s
		return padded.to(self.device)

	def build_child_log_probability_fn(self) -> Callable[[List[sbs.State]], List[np.ndarray]]:
		"""Return SBS's child_log_probability_fn for this model.

		Given a list of states (partial sequences), returns a list of per-state
		next-token log-probability vectors (as numpy arrays). Applies temperature,
		optional top-k truncation, masks PAD, and uses an attention mask.
		"""
		@torch.no_grad()
		def child_log_probability_fn(states: List[torch.Tensor]) -> List[np.ndarray]:
			"""Compute batched next-token log-probabilities for given states.
			    # 1. Take current partial sequences (states)
				# 2. Run them through Hyformer model
				# 3. Get logits for next tokens
				# 4. Apply temperature/top_k filtering
				# 5. Convert to log probabilities
				# 6. Return numpy arrays for SBS
				"""
			if len(states) == 0:
				return []
			self.model.eval()
			batch = self._collate_states(states)
			# Build attention mask (1 for tokens, 0 for PAD) if PAD is defined
			if self.pad_id is not None:
				attn = (batch != int(self.pad_id)).long()
			else:
				attn = None
			outputs = self.model(input_ids=batch, attention_mask=attn, next_token_only=True, task='generation')
			logits = outputs['logits_generation'][:, -1, :]
			if self.temperature != 1.0:
				logits = logits / max(1e-8, self.temperature)
			if self.top_k is not None and self.top_k > 0 and self.top_k < logits.size(-1):
				v, idx = torch.topk(logits, k=self.top_k, dim=-1)
				masked = torch.full_like(logits, float('-inf'))
				masked.scatter_(1, idx, v)
				logits = masked
			log_probs = F.log_softmax(logits, dim=-1)
			if self.pad_id is not None and 0 <= int(self.pad_id) < log_probs.size(-1):
				log_probs[:, int(self.pad_id)] = float('-inf')
			ret: List[np.ndarray] = []
			for i in range(log_probs.size(0)):
				ret.append(log_probs[i].detach().cpu().numpy())
			return ret
		return child_log_probability_fn

	def build_child_transition_fn(self) -> Callable[[List[Tuple[sbs.State, int]]], List[Tuple[sbs.State, bool]]]:
		"""Return SBS's child_transition_fn for this model.

		Maps (state, action_id) → (next_state, is_leaf) by appending the token
		and checking EOS or max length to decide leaf termination.
		"""
		def child_transition_fn(state_action_pairs: List[Tuple[torch.Tensor, int]]) -> List[Tuple[torch.Tensor, bool]]:
			"""Append action to each state and flag whether it has become a leaf.
			    # 1. Take (current_sequence, chosen_token_id)
				# 2. Append token to sequence
				# 3. Check if sequence is complete (leaf)
				# 4. Return (new_sequence, is_complete)
				"""
			results: List[Tuple[torch.Tensor, bool]] = []
			for prefix, action_idx in state_action_pairs:
				action = int(action_idx)
				new_ids = torch.cat([prefix, torch.tensor([action], dtype=torch.long)], dim=0)
				is_leaf = (self.eos_id is not None and action == int(self.eos_id)) or (new_ids.size(0) >= self.max_new_tokens)
				results.append((new_ids, bool(is_leaf)))
			return results
		return child_transition_fn

	def build_leaf_evaluation_fn(self, oracle: Optional[Callable[[str], float]] = None, invalid_penalty: float = 0.0):
		"""Return a callable that scores leaves with an oracle.

		If no oracle is provided, returns zeros. Otherwise, decodes token ids to
		SMILES, checks validity, computes the oracle score (e.g., QED), and returns
		a float (or list of floats for batched input). Non-finite/invalid are 0.0.
		"""
		# Return a function that accepts EITHER a single state OR a list of states.
		# - Single state -> float
		# - List[states] -> List[float]
		if oracle is None:
			def _zeros(arg):
				if isinstance(arg, (list, tuple)):
					return [0.0 for _ in arg]
				return 0.0
			return _zeros

		def _decode_many(states: List[torch.Tensor]) -> List[str]:
			if not states:
				return []
			max_len = max(int(x.size(0)) for x in states)
			pad_id = getattr(self.tokenizer, 'pad_token_id', 0) or 0
			batch = torch.full((len(states), max_len), int(pad_id), dtype=torch.long)
			for i, s in enumerate(states):
				batch[i, : s.size(0)] = s
			# Provide attention mask for decoding if supported; tokenizer.decode here uses underlying tokenizer
			return self.tokenizer.decode(batch, skip_special_tokens=True)

		def leaf_eval_fn(arg):
			# Normalize to list
			# 1. Decode token sequences to SMILES strings
			# 2. Check SMILES validity
			# 3. Compute oracle score (QED, etc.)
			# 4. Return score (higher = better for SBS) 
	
			is_list = isinstance(arg, (list, tuple))
			states = list(arg) if is_list else [arg]
			# Defensive: ensure all are 1D LongTensor
			proc: List[torch.Tensor] = []
			for st in states:
				if isinstance(st, torch.Tensor):
					proc.append(st)
				else:
					# If a wrapper slipped through, try to extract tensor
					# Otherwise, skip with zero score placeholder
					try:
						proc.append(torch.as_tensor(st, dtype=torch.long))
					except Exception:
						proc.append(torch.tensor([], dtype=torch.long))
			try:
				texts = _decode_many(proc)
				vals = []
				for t in texts:
					try:
						# Always delegate scoring to oracle; apply invalid_penalty if invalid/non-finite
						v = oracle(t)
						vf = float(v) if v is not None else invalid_penalty
						if (not is_valid(t)) or (not np.isfinite(vf)):
							vf = invalid_penalty
						vals.append(vf)
					except Exception:
						vals.append(invalid_penalty)
			except Exception:
				vals = [invalid_penalty] * len(proc)
			return vals if is_list else vals[0]

		return leaf_eval_fn

"""
How sequences are built in inc_sbs:
SBS repeatedly calls:
child_log_probability_fn → get logits from Hyformer
child_transition_fn → append the chosen token to the prefix (build the SMILES)
"""

def run_incremental_gumbeldore(*, model, tokenizer, device: torch.device, oracle: Optional[Callable[[str], float]] = None, beam_width: int = 32, num_rounds: int = 4, max_new_tokens: int = 128, temperature: float = 1.0, top_k: Optional[int] = 12, advantage_constant: float = 1.0, min_max_normalize_advantage: bool = False, expected_value_use_simple_mean: bool = False, use_pure_outcomes: bool = False, normalize_advantage_by_visit_count: bool = False, perform_first_round_deterministic: bool = False, min_nucleus_top_p: float = 1.0, initial_batch: Optional[int] = None, invalid_penalty: float = -1.1) -> List[List[sbs.BeamLeaf]]:
	"""Run Incremental SBS with Gumbeldore policy updates.

	Constructs SBS-wrapped callables from the adapter, initializes root states
	from the generation prefix, performs num_rounds of SBS with beam_width, and
	returns sampled leaves per root. Uses leaf_eval_fn to compute advantages.
	"""
	adapter = HyformerSBSAdapter(model=model, tokenizer=tokenizer, device=device, temperature=temperature, top_k=top_k, max_new_tokens=max_new_tokens)
	child_log_probability_fn = adapter.build_child_log_probability_fn()
	child_transition_fn = adapter.build_child_transition_fn()
	leaf_evaluation_fn = adapter.build_leaf_evaluation_fn(oracle, invalid_penalty=invalid_penalty)
	prefix = tokenizer.generation_prefix
	if isinstance(prefix, int):
		root = torch.tensor([prefix], dtype=torch.long)
	else:
		root = torch.tensor(prefix, dtype=torch.long)
	batch_n = int(initial_batch) if initial_batch is not None else 1
	initial_states: List[torch.Tensor] = [root.clone() for _ in range(batch_n)]
	inc = IncrementalSBS(initial_states=initial_states, child_log_probability_fn=child_log_probability_fn, child_transition_fn=child_transition_fn, leaf_evaluation_fn=leaf_evaluation_fn, memory_aggressive=False)
	result = inc.perform_incremental_sbs(beam_width=beam_width, num_rounds=num_rounds, log_prob_update_type="gumbeldore", advantage_constant=advantage_constant, min_max_normalize_advantage=min_max_normalize_advantage, expected_value_use_simple_mean=expected_value_use_simple_mean, use_pure_outcomes=use_pure_outcomes, normalize_advantage_by_visit_count=normalize_advantage_by_visit_count, perform_first_round_deterministic=perform_first_round_deterministic, min_nucleus_top_p=min_nucleus_top_p, return_round_info=True)
	return result


def _setup_model_and_tokenizer(model_config_path: str, model_ckpt_path: str, tokenizer_config_path: str, device: torch.device):
	"""Load model and tokenizer, move model to device, set eval mode."""
	model_config = ModelConfig.from_config_file(model_config_path)
	# Follow qed_samplingv2: instantiate downstream model explicitly
	from hyformer.models.hyformer import HyformerForDownstreamPrediction
	model = HyformerForDownstreamPrediction.from_config(
		model_config,
		downstream_task="regression",
		num_tasks=1,
	)
	model.load_pretrained(model_ckpt_path)
	model.to(device)
	model.eval()
	tokenizer_config = TokenizerConfig.from_config_file(tokenizer_config_path)
	tokenizer = AutoTokenizer.from_config(tokenizer_config)
	return model, tokenizer


def _decode_list(tokenizer, seqs: List[torch.Tensor]) -> List[str]:
	"""Batch-decode a list of 1D id tensors to SMILES strings."""
	if not seqs:
		return []
	max_len = max(int(x.size(0)) for x in seqs)
	pad_id = getattr(tokenizer, 'pad_token_id', 0) or 0
	batch = torch.full((len(seqs), max_len), int(pad_id), dtype=torch.long)
	for i, s in enumerate(seqs):
		batch[i, : s.size(0)] = s
	return tokenizer.decode(batch, skip_special_tokens=True)


def main():
	"""CLI entrypoint: parses args, runs naive or inc_sbs, optional CSV write."""
	import argparse
	parser = argparse.ArgumentParser(description="Hyformer + Incremental SBS (Gumbeldore)")
	parser.add_argument('--model_config', type=str, required=True)
	parser.add_argument('--model_ckpt', type=str, required=True)
	parser.add_argument('--tokenizer_config', type=str, required=True)
	parser.add_argument('--device', type=str, default=None)
	parser.add_argument('--method', type=str, choices=['inc_sbs'], default='inc_sbs')
	parser.add_argument('--num_samples', type=int, default=16)
	parser.add_argument('--max_new_tokens', type=int, default=128)
	parser.add_argument('--temperature', type=float, default=1.0)
	parser.add_argument('--top_k', type=int, default=0)
	parser.add_argument('--beam_width', type=int, default=32)
	parser.add_argument('--num_rounds', type=int, default=4)
	parser.add_argument('--advantage_constant', type=float, default=1.0)
	parser.add_argument('--min_max_norm', action='store_true')
	parser.add_argument('--simple_mean', action='store_true')
	parser.add_argument('--pure_outcomes', action='store_true')
	parser.add_argument('--norm_by_visits', action='store_true')
	parser.add_argument('--deterministic_first', action='store_true')
	parser.add_argument('--min_top_p', type=float, default=1.0)
	parser.add_argument('--target_name', type=str, choices=['qed', 'sa', 'logp'], default='qed', help='Property to use as target (qed, sa, or logp)')
	parser.add_argument('--oracle_source', type=str, choices=['rdkit', 'prediction'], default='rdkit', help='Use RDKit oracle or Hyformer prediction head as oracle')
	parser.add_argument('--target_value', type=float, default=None, help='Optional target value for property; used to bias scores toward closeness')
	parser.add_argument('--pred_mean', type=float, default=None, help='Optional mean to inverse-transform predicted values: y = y_hat*std + mean')
	parser.add_argument('--pred_std', type=float, default=None, help='Optional std to inverse-transform predicted values')
	parser.add_argument('--invalid_penalty', type=float, default=-1.1, help='Score assigned to invalid/non-finite leaves (strictly worse than any valid)')
	parser.add_argument('--loss_function', type=str, choices=['1minusAbsoluteError', 'negativeAbsoluteError'], default='1minusAbsoluteError', help='Loss function to use: 1minusAbsoluteError (good for 0-1 range) or negativeAbsoluteError (good for wider ranges)')
	parser.add_argument('--out_csv', type=str, default=None, help='Optional path to save CSV with columns: batch,smiles')
	parser.add_argument('--out_csv_selected', type=str, default=None, help='Optional path to save selected molecules.')
	args = parser.parse_args()
	device = torch.device(args.device) if args.device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model, tokenizer = _setup_model_and_tokenizer(args.model_config, args.model_ckpt, args.tokenizer_config, device)
	top_k = args.top_k if args.top_k > 0 else None
	
	# Build oracle. If target_value is provided, use a target-aware wrapper that maximizes closeness.
	base_oracle = None

	# Optional: Hyformer prediction head as oracle
	@torch.no_grad()
	def _predict_property_normalized(smiles: str) -> float:
		"""Get normalized prediction (raw model output)."""
		inputs = tokenizer([smiles], task='prediction')
		inputs = inputs.to(device)
		pred = model.predict(**inputs).detach().cpu().numpy()
		return float(pred.squeeze())
	
	@torch.no_grad()
	def _predict_property_denormalized(smiles: str) -> float:
		"""Get denormalized prediction (actual property scale)."""
		val_z = _predict_property_normalized(smiles)
		if args.pred_mean is not None and args.pred_std is not None and np.isfinite(val_z):
			return val_z * float(args.pred_std) + float(args.pred_mean)
		return val_z


	# RDKit oracles for all properties (independent of oracle_source for logging)
	from rdkit import Chem
	from rdkit.Chem import AllChem, Crippen
	from rdkit.Chem.QED import qed
	from hyformer.utils.properties.smiles.sascorer import compute_sa_score
	
	def get_qed(smiles: str) -> float:
		try:
			mol = Chem.MolFromSmiles(smiles)
			try:
				AllChem.Kekulize(mol, clearAromaticFlags=True)
			except:
				pass
			mol.UpdatePropertyCache(strict=False)
			return float(qed(mol))
		except:
			return float("nan")
	
	def get_sa(smiles: str) -> float:
		try:
			mol = Chem.MolFromSmiles(smiles)
			try:
				AllChem.Kekulize(mol, clearAromaticFlags=True)
			except:
				pass
			mol.UpdatePropertyCache(strict=False)
			# Use the same normalization as notebook: (10 - sa) / 9
			# This gives 0.0-1.0 range where lower = easier to synthesize
			return float(compute_sa_score(mol))
		except:
			return float('nan')
	
	def get_logp(smiles: str) -> float:
		try:
			mol = Chem.MolFromSmiles(smiles)
			try:
				AllChem.Kekulize(mol, clearAromaticFlags=True)
			except:
				pass
			mol.UpdatePropertyCache(strict=False)
			return float(Crippen.MolLogP(mol))
		except:
			return float('nan')
	
	_rdkit_oracles = {
		'qed': get_qed,
		'sa': get_sa,
		'logp': get_logp
	}

	@torch.no_grad()
	def _property_rdkit(smiles: str, property_name: str) -> float:
		try:
			if not is_valid(smiles):
				return float('nan')
			oracle_func = _rdkit_oracles[property_name]
			val = float(oracle_func(smiles))
			return val if np.isfinite(val) else float('nan')
		except Exception:
			return float('nan')

	if args.oracle_source == 'prediction':
		base_oracle = _predict_property_denormalized
	elif args.target_name:
		if args.target_name == 'qed':
			base_oracle = get_qed
		elif args.target_name == 'sa':
			base_oracle = get_sa
		elif args.target_name == 'logp':
			base_oracle = get_logp
	oracle = base_oracle
	if base_oracle is not None and args.target_value is not None:
		target_for_oracle = float(args.target_value)
		def oracle_target(smiles: str) -> float:
				q = base_oracle(smiles)
				qf = float(q)
				abs_error = abs(qf - target_for_oracle)
				
				# Apply selected loss function
				if args.loss_function == '1minusAbsoluteError':
					# Higher is better: 1 - |predicted - target| (good for 0-1 range)
					return 1.0 - abs_error
				elif args.loss_function == 'negativeAbsoluteError':
					# Higher is better: -|predicted - target| (good for wider ranges)
					return -abs_error
				else:
					raise ValueError(f"Unknown loss function: {args.loss_function}")
		oracle = oracle_target
		
	def write_csv(rows, include_qed: bool):
		if not args.out_csv:
			return
		import os, csv
		os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
		file_exists = os.path.exists(args.out_csv)
		fieldnames = ['batch', 'smiles'] + (['qed'] if include_qed else [])
		with open(args.out_csv, 'a', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			if not file_exists:
				writer.writeheader()
			for r in rows:
				writer.writerow(r)


	if args.method == 'inc_sbs':
		total_generated = 0
		valid_all_generated = 0
		seen_smiles = set()
		all_rows = []  # Collect all rows for CSV writing
		# Run SBS with all rounds in one call for efficiency
		res = run_incremental_gumbeldore(model=model, tokenizer=tokenizer, device=device, oracle=oracle, beam_width=args.beam_width, num_rounds=args.num_rounds, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=top_k, advantage_constant=args.advantage_constant, min_max_normalize_advantage=args.min_max_norm, expected_value_use_simple_mean=args.simple_mean, use_pure_outcomes=args.pure_outcomes, normalize_advantage_by_visit_count=args.norm_by_visits, perform_first_round_deterministic=args.deterministic_first, min_nucleus_top_p=args.min_top_p, initial_batch=1, invalid_penalty=float(args.invalid_penalty))
		leaves_per_root, round_info_batch = res
		leaves = leaves_per_root[0]
		round_info = round_info_batch[0]
		seqs = [leaf.state for leaf in leaves]
		decoded = _decode_list(tokenizer, seqs)
		rows = []
		for i, (s, leaf, (rnd, rank)) in enumerate(zip(decoded, leaves, round_info)):
			print(s)
			property_score = None
			property_real = None
			property_pred_z = None
			property_pred = None
			# Debug: compute token-length diagnostics relative to generation prefix and EOS
			# Determine prefix length
			gen_prefix = tokenizer.generation_prefix
			if isinstance(gen_prefix, list):
				_prefix_len = len(gen_prefix)
			elif isinstance(gen_prefix, int):
				_prefix_len = 1
			else:
				_p = tokenizer([gen_prefix], task='generation')['input_ids']
				_prefix_len = int(_p.size(1)) if hasattr(_p, 'size') else len(_p[0])
			_eos_id = getattr(tokenizer, 'sep_token_id', None)
			_tokens_1d = leaf.state
			_end_pos = int(_tokens_1d.size(0))
			if _eos_id is not None:
				_where = (_tokens_1d == int(_eos_id)).nonzero(as_tuple=False)
				if _where.numel() > 0:
					_end_pos = int(_where[0, 0].item())
			_tokens_after_prefix = max(0, _end_pos - max(1, _prefix_len))
			
			# Get RDKit property values (real values)
			property_real = _property_rdkit(s, args.target_name)
			
			# Get model predictions
			try:
				property_pred_z = float(_predict_property_normalized(s))  # Normalized (raw model output)
				if not np.isfinite(property_pred_z):
					property_pred_z = float('nan')
			except Exception:
				property_pred_z = float('nan')
			
			try:
				property_pred = float(_predict_property_denormalized(s))  # Denormalized (actual property scale)
				if not np.isfinite(property_pred):
					property_pred = float('nan')
			except Exception:
				property_pred = float('nan')
			
			# Calculate oracle score (SBS objective)
			if oracle is not None:
				try:
					property_score = float(oracle(s))
				except Exception:
					property_score = float('nan')
			
			# Calculate the actual top_p value used in this round BEFORE top-p log-prob calc
			if args.num_rounds == 1:
				top_p_used = 1.0
			else:
				top_p_used = (1 - rnd / (args.num_rounds - 1.0)) * args.min_top_p + 1.0 * (rnd / (args.num_rounds - 1.0))

			
			
			row = {
				'batch': i,
				'round': int(rnd),
				'leaf_rank': int(rank),
				'smiles': s,
				'is_valid': bool(is_valid(s)),
				'gumbel': float(leaf.gumbel), #The Gumbel-perturbed score used by SBS to rank leaves during "without-replacement" sampling.
				'advantage_constant': args.advantage_constant, #The advantage constant used for policy updates
				'tokens_after_prefix': int(_tokens_after_prefix),
			}

			
			# Add property values
			row[f'{args.target_name}_real'] = property_real  # Real property from RDKit
			row[f'{args.target_name}_pred_z'] = property_pred_z  # Normalized prediction (raw model output)
			row[f'{args.target_name}_pred'] = property_pred  # Denormalized prediction (actual property scale)
			if property_score is not None:
				row[f'{args.target_name}_score'] = property_score  # Oracle score (SBS objective)
			
			
			
			# Add additional debugging parameters
			row['temperature'] = args.temperature
			row['top_k'] = args.top_k if args.top_k > 0 else 0
			row['beam_width'] = args.beam_width
			row['min_top_p'] = args.min_top_p
			
			row['top_p_used'] = top_p_used
			
			rows.append(row)
		
		# Add rows to the collection
		all_rows.extend(rows)
		# accumulate validity on all generated
		total_generated += int(len(rows))
		for r in rows:
			try:
				if bool(r.get('is_valid', False)):
					valid_all_generated += 1
			except Exception:
				pass
		
		# Write extended schema after all rounds
		def write_csv_ext(rows):
			if not args.out_csv:
				return
			import os, csv
			os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
			file_exists = os.path.exists(args.out_csv)
			base_cols = ['batch', 'round', 'leaf_rank', 'smiles', 'is_valid', 'gumbel', 'advantage_constant', 'tokens_after_prefix']
			base_cols.extend([f'{args.target_name}_real', f'{args.target_name}_pred_z', f'{args.target_name}_pred'])  # Always include these
			if oracle is not None:
				base_cols.append(f'{args.target_name}_score')
			# Add debugging columns
			base_cols.extend(['temperature', 'top_k', 'beam_width', 'min_top_p', 'top_p_used'])
			fieldnames = base_cols
			with open(args.out_csv, 'w', newline='') as f:  # Changed from 'a' to 'w' to write fresh file
				writer = csv.DictWriter(f, fieldnames=fieldnames)
				writer.writeheader()  # Always write header for fresh file
				for r in rows:
					writer.writerow(r)
		
		# Write all collected data to CSV
		write_csv_ext(all_rows)
		
		# Print final statistics
		if total_generated > 0:
			valid_pct = 100.0 * float(valid_all_generated) / float(total_generated)
			print(f"Validity (all generated): {valid_pct:.2f}% ({valid_all_generated}/{total_generated})")
		else:
			print("Validity (all generated): N/A (no rows)")
		print(f"Completed {args.num_rounds} rounds of sampling.")


if __name__ == '__main__':
	main()


"""
property_real: RDKit property value (NaN if invalid).
property_pred_z: Hyformer prediction head's property (normalized), included when --oracle_source prediction.
property_pred: Hyformer prediction head's property (inverse-transformed if you pass --pred_mean/--pred_std), included when --oracle_source prediction.
property_score: the actual SBS objective (−abs or −square), from the oracle you chose.
abs_error/sq_error: vs target_value when provided.
Invalid handling is centralized in leaf_eval_fn via --invalid_penalty (default −1.1)

Supported properties: qed, sa, logp
"""