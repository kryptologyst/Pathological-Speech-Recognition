"""Evaluation metrics for pathological speech recognition."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import editdistance
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class PathologicalSpeechMetrics:
    """Metrics specific to pathological speech recognition."""
    
    def __init__(self, vocab_size: int = 32):
        """Initialize metrics.
        
        Args:
            vocab_size: Vocabulary size.
        """
        self.vocab_size = vocab_size
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.total_wer = 0.0
        self.total_cer = 0.0
        self.total_samples = 0
        self.total_words = 0
        self.total_chars = 0
        self.total_substitutions = 0
        self.total_insertions = 0
        self.total_deletions = 0
        
        # Pathological speech specific metrics
        self.intelligibility_scores = []
        self.fluency_scores = []
        self.articulation_scores = []
        
        # Per-condition metrics
        self.condition_metrics = defaultdict(lambda: {
            "wer": 0.0,
            "cer": 0.0,
            "samples": 0,
            "words": 0,
            "chars": 0,
        })
    
    def compute_wer(self, reference: str, hypothesis: str) -> float:
        """Compute Word Error Rate.
        
        Args:
            reference: Reference text.
            hypothesis: Hypothesis text.
            
        Returns:
            WER value.
        """
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        if len(ref_words) == 0:
            return 1.0 if len(hyp_words) > 0 else 0.0
        
        # Compute edit distance
        distance = editdistance.eval(ref_words, hyp_words)
        return distance / len(ref_words)
    
    def compute_cer(self, reference: str, hypothesis: str) -> float:
        """Compute Character Error Rate.
        
        Args:
            reference: Reference text.
            hypothesis: Hypothesis text.
            
        Returns:
            CER value.
        """
        ref_chars = list(reference.lower().replace(" ", ""))
        hyp_chars = list(hypothesis.lower().replace(" ", ""))
        
        if len(ref_chars) == 0:
            return 1.0 if len(hyp_chars) > 0 else 0.0
        
        # Compute edit distance
        distance = editdistance.eval(ref_chars, hyp_chars)
        return distance / len(ref_chars)
    
    def compute_intelligibility_score(self, reference: str, hypothesis: str) -> float:
        """Compute intelligibility score for pathological speech.
        
        Args:
            reference: Reference text.
            hypothesis: Hypothesis text.
            
        Returns:
            Intelligibility score (0-1, higher is better).
        """
        # Simple intelligibility based on WER
        wer = self.compute_wer(reference, hypothesis)
        
        # Convert WER to intelligibility score
        # WER 0.0 -> 1.0, WER 1.0 -> 0.0
        intelligibility = max(0.0, 1.0 - wer)
        
        return intelligibility
    
    def compute_fluency_score(self, hypothesis: str) -> float:
        """Compute fluency score based on speech characteristics.
        
        Args:
            hypothesis: Hypothesis text.
            
        Returns:
            Fluency score (0-1, higher is better).
        """
        if not hypothesis:
            return 0.0
        
        words = hypothesis.split()
        if len(words) == 0:
            return 0.0
        
        # Simple fluency metrics
        avg_word_length = np.mean([len(word) for word in words])
        word_count = len(words)
        
        # Normalize metrics
        avg_word_length_score = min(1.0, avg_word_length / 6.0)  # Assume 6 chars is good
        word_count_score = min(1.0, word_count / 10.0)  # Assume 10 words is good
        
        # Combine metrics
        fluency = (avg_word_length_score + word_count_score) / 2.0
        
        return fluency
    
    def compute_articulation_score(self, reference: str, hypothesis: str) -> float:
        """Compute articulation score for pathological speech.
        
        Args:
            reference: Reference text.
            hypothesis: Hypothesis text.
            
        Returns:
            Articulation score (0-1, higher is better).
        """
        # Based on character-level accuracy
        cer = self.compute_cer(reference, hypothesis)
        articulation = max(0.0, 1.0 - cer)
        
        return articulation
    
    def update(
        self,
        references: List[str],
        hypotheses: List[str],
        conditions: Optional[List[str]] = None,
    ) -> None:
        """Update metrics with new predictions.
        
        Args:
            references: List of reference texts.
            hypotheses: List of hypothesis texts.
            conditions: List of pathological conditions (optional).
        """
        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
            # Compute basic metrics
            wer = self.compute_wer(ref, hyp)
            cer = self.compute_cer(ref, hyp)
            
            # Update totals
            self.total_wer += wer
            self.total_cer += cer
            self.total_samples += 1
            
            ref_words = ref.split()
            ref_chars = list(ref.replace(" ", ""))
            self.total_words += len(ref_words)
            self.total_chars += len(ref_chars)
            
            # Compute pathological speech specific metrics
            intelligibility = self.compute_intelligibility_score(ref, hyp)
            fluency = self.compute_fluency_score(hyp)
            articulation = self.compute_articulation_score(ref, hyp)
            
            self.intelligibility_scores.append(intelligibility)
            self.fluency_scores.append(fluency)
            self.articulation_scores.append(articulation)
            
            # Update per-condition metrics
            if conditions and i < len(conditions):
                condition = conditions[i]
                self.condition_metrics[condition]["wer"] += wer
                self.condition_metrics[condition]["cer"] += cer
                self.condition_metrics[condition]["samples"] += 1
                self.condition_metrics[condition]["words"] += len(ref_words)
                self.condition_metrics[condition]["chars"] += len(ref_chars)
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics.
        
        Returns:
            Dictionary of computed metrics.
        """
        if self.total_samples == 0:
            return {}
        
        metrics = {
            "wer": self.total_wer / self.total_samples,
            "cer": self.total_cer / self.total_samples,
            "intelligibility": np.mean(self.intelligibility_scores) if self.intelligibility_scores else 0.0,
            "fluency": np.mean(self.fluency_scores) if self.fluency_scores else 0.0,
            "articulation": np.mean(self.articulation_scores) if self.articulation_scores else 0.0,
        }
        
        # Add per-condition metrics
        for condition, cond_metrics in self.condition_metrics.items():
            if cond_metrics["samples"] > 0:
                metrics[f"{condition}_wer"] = cond_metrics["wer"] / cond_metrics["samples"]
                metrics[f"{condition}_cer"] = cond_metrics["cer"] / cond_metrics["samples"]
        
        return metrics


class ConfidenceCalibration:
    """Confidence calibration for pathological speech recognition."""
    
    def __init__(self):
        """Initialize confidence calibration."""
        self.predictions = []
        self.confidences = []
        self.correct = []
    
    def update(self, predictions: List[str], confidences: List[float], references: List[str]) -> None:
        """Update calibration data.
        
        Args:
            predictions: Model predictions.
            confidences: Confidence scores.
            references: Reference texts.
        """
        for pred, conf, ref in zip(predictions, confidences, references):
            # Simple correctness check (exact match)
            correct = pred.lower().strip() == ref.lower().strip()
            
            self.predictions.append(pred)
            self.confidences.append(conf)
            self.correct.append(correct)
    
    def compute_ece(self, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error.
        
        Args:
            n_bins: Number of bins for calibration.
            
        Returns:
            ECE value.
        """
        if not self.confidences:
            return 0.0
        
        # Sort by confidence
        sorted_indices = np.argsort(self.confidences)
        sorted_confidences = np.array(self.confidences)[sorted_indices]
        sorted_correct = np.array(self.correct)[sorted_indices]
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (sorted_confidences > bin_lower) & (sorted_confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = sorted_correct[in_bin].mean()
                avg_confidence_in_bin = sorted_confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def compute_reliability_diagram(self, n_bins: int = 10) -> Dict[str, List[float]]:
        """Compute reliability diagram data.
        
        Args:
            n_bins: Number of bins.
            
        Returns:
            Dictionary with bin centers, accuracies, and confidences.
        """
        if not self.confidences:
            return {"bin_centers": [], "accuracies": [], "confidences": []}
        
        # Sort by confidence
        sorted_indices = np.argsort(self.confidences)
        sorted_confidences = np.array(self.confidences)[sorted_indices]
        sorted_correct = np.array(self.correct)[sorted_indices]
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        accuracies = []
        confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (sorted_confidences > bin_lower) & (sorted_confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                accuracies.append(sorted_correct[in_bin].mean())
                confidences.append(sorted_confidences[in_bin].mean())
        
        return {
            "bin_centers": bin_centers,
            "accuracies": accuracies,
            "confidences": confidences,
        }
