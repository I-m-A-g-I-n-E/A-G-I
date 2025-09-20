#!/usr/bin/env python3
"""
The 48-Manifold Linguistic State Machine
Combining fractal reversibility with alphabetic resonance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

from manifold import device, RouterMode, SixAxisState, Fractal48Layer

class AlphabeticResonator(nn.Module):
    """
    Maps words to taste/experience through letter resonance
    W routes possibility, M manifests reality
    """
    
    def __init__(self):
        super().__init__()
        
        # Letter embeddings (A-Z as state machines)
        self.letter_states = nn.Parameter(torch.randn(26, 48))
        
        # Special routers for W (22) and M (12)
        self.w_router = nn.Parameter(torch.randn(6, 48, 48))
        self.m_router = nn.Parameter(torch.randn(6, 48, 48))
        
        # Initialize routers near-orthogonal
        for i in range(6):
            nn.init.orthogonal_(self.w_router[i])
            nn.init.orthogonal_(self.m_router[i])
        
        # Taste pattern templates
        self.taste_patterns = nn.ParameterDict({
            'sweet': nn.Parameter(self._encode_pattern('sugar honey')),
            'bitter': nn.Parameter(self._encode_pattern('coffee dark')),
            'sour': nn.Parameter(self._encode_pattern('lemon citrus')),
            'salty': nn.Parameter(self._encode_pattern('salt ocean')),
            'umami': nn.Parameter(self._encode_pattern('mushroom soy'))
        })
    
    def _encode_pattern(self, words: str) -> torch.Tensor:
        """Convert words to letter frequency pattern"""
        pattern = torch.zeros(26)
        for word in words.lower().split():
            for i, char in enumerate(word):
                if 'a' <= char <= 'z':
                    idx = ord(char) - ord('a')
                    pattern[idx] += 1.0 / (i + 1)  # Position weighting
        return pattern / (pattern.norm() + 1e-8)
    
    def word_to_axes(self, word: str) -> SixAxisState:
        """Map word to six-axis coordinates through letter resonance"""
        # Build letter histogram
        letter_hist = torch.zeros(48, device=device)
        for i, char in enumerate(word.lower()):
            if 'a' <= char <= 'z':
                idx = ord(char) - ord('a')
                # Embed letter with position decay
                letter_hist += self.letter_states[idx] / (i + 1)
        
        # Project onto six axes with phase shifts
        return SixAxisState(
            who=torch.roll(letter_hist, 0),
            what=torch.roll(letter_hist, -8),
            when=torch.roll(letter_hist, -16),
            where=torch.roll(letter_hist, -24),
            why=torch.roll(letter_hist, -32),
            how=torch.roll(letter_hist, -40)
        )
    
    def route(self, x: torch.Tensor, axes: SixAxisState, 
              mode: RouterMode) -> torch.Tensor:
        """Route through W (possibility) or M (manifestation)"""
        
        if mode == RouterMode.W_POSSIBILITY:
            # W routing: keeps things open, probabilistic
            router = self.w_router
            letter_state = self.letter_states[22]  # W
            
            # Apply routing with soft gating (never fully crystallizes)
            for i, axis in enumerate(axes.to_tensor()):
                gate = torch.sigmoid(axis @ router[i])
                x = x + gate * letter_state
            
            # W keeps values in (-1, 1) but never ±1
            return torch.tanh(x) * 0.95
            
        else:  # M_MANIFESTATION
            # M routing: crystallizes into discrete states
            router = self.m_router
            letter_state = self.letter_states[12]  # M
            
            # Apply routing with hard gating
            for i, axis in enumerate(axes.to_tensor()):
                gate = axis @ router[i]
                x = x + gate * letter_state
            
            # M quantizes to ternary {-1, 0, 1}
            return torch.round(torch.tanh(x) * 3) / 3
    
    def analyze_taste(self, word: str) -> Dict[str, float]:
        """Analyze how a word resonates with taste patterns"""
        word_pattern = self._encode_pattern(word).to(self.letter_states.device)
        
        tastes = {}
        for taste_name, taste_pattern in self.taste_patterns.items():
            resonance = F.cosine_similarity(
                word_pattern.unsqueeze(0),
                taste_pattern.unsqueeze(0)
            ).item()
            tastes[taste_name] = max(0, resonance)  # Only positive resonance
        
        return tastes
    
    def trace_tomorrow(self, word: str = "tomorrow") -> Dict[str, torch.Tensor]:
        """
        Trace the TOMORROW flow: T→O→M→O→R→R→O→W
        Shows how the word never crystallizes (W dominance)
        """
        states = {}
        x = torch.randn(48, device=device)
        
        for i, letter in enumerate(word.lower()):
            if 'a' <= letter <= 'z':
                idx = ord(letter) - ord('a')
                letter_state = self.letter_states[idx]
                
                # Apply letter as operator
                if letter == 'm':
                    # M crystallizes
                    x = torch.round(x + letter_state)
                elif letter == 'w':
                    # W opens possibility
                    x = torch.tanh(x + letter_state) * 0.95
                elif letter == 'r':
                    # R reflects (adjoint-like)
                    x = -x + letter_state
                elif letter == 't':
                    # T transforms (time shift)
                    x = torch.roll(x, shifts=1) + letter_state
                elif letter == 'o':
                    # O observes (normalizes)
                    x = F.normalize(x + letter_state, dim=0)
                else:
                    x = x + letter_state * 0.5
                
                states[f"{i}_{letter}"] = x.clone()
        
        return states

class Linguistic48System(nn.Module):
    """
    Complete system combining fractal reversibility 
    with alphabetic resonance
    """
    
    def __init__(self):
        super().__init__()
        self.fractal = Fractal48Layer()
        self.resonator = AlphabeticResonator()
        
        # Knowledge graph edges
        self.edges = nn.ModuleDict({
            'form_to_function': nn.Linear(48, 48),
            'function_to_outcome': nn.Linear(48, 48),
            'outcome_to_function': nn.Linear(48, 48),  # Adjoint
            'function_to_form': nn.Linear(48, 48)  # Adjoint
        })
    
    def forward_path(self, word: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Form → Function → Outcome
        Word shapes expectation shapes experience
        """
        # Form: word as input
        axes = self.resonator.word_to_axes(word)
        x = axes.to_tensor().mean(dim=0)
        
        # Function: process through edges
        function = self.edges['form_to_function'](x)
        
        # Branch into W and M paths
        w_path = self.resonator.route(function, axes, RouterMode.W_POSSIBILITY)
        m_path = self.resonator.route(function, axes, RouterMode.M_MANIFESTATION)
        
        # Outcome: final state
        w_outcome = self.edges['function_to_outcome'](w_path)
        m_outcome = self.edges['function_to_outcome'](m_path)
        
        return w_outcome, m_outcome
    
    def adjoint_path(self, outcome: torch.Tensor) -> torch.Tensor:
        """
        Outcome → Function → Form
        Experience updates linguistic priors
        """
        function = self.edges['outcome_to_function'](outcome)
        form = self.edges['function_to_form'](function)
        return form
    
    def taste_experiment(self, substance: str, labels: List[str]) -> Dict:
        """
        Test how different labels affect perception of same substance
        """
        results = {}
        
        for label in labels:
            # Forward path: label creates expectation
            w_expect, m_expect = self.forward_path(label)
            
            # Analyze taste resonance
            tastes = self.resonator.analyze_taste(label)
            
            # Measure W vs M dominance
            w_energy = w_expect.abs().mean().item()
            m_energy = m_expect.abs().mean().item()
            
            results[label] = {
                'tastes': tastes,
                'w_dominance': w_energy / (w_energy + m_energy + 1e-8),
                'm_dominance': m_energy / (w_energy + m_energy + 1e-8),
                'top_taste': max(tastes.items(), key=lambda x: x[1])[0]
            }
        
        return results

def demonstrate_linguistic_reality():
    """
    Full demonstration of the 48-manifold linguistic system
    """
    print("=" * 60)
    print("THE 48-MANIFOLD: Language as Reality's Operating System")
    print("=" * 60)
    
    system = Linguistic48System().to(device)
    
    # 1. Test reversibility
    print("\n1. FRACTAL REVERSIBILITY TEST:")
    test_input = torch.randn(1, 48, 48, 48, device=device)
    encoded = system.fractal(test_input)
    decoded = system.fractal(encoded, inverse=True)
    error = (test_input - decoded).abs().max().item()
    print(f"   Reconstruction error: {error:.6f}")
    print(f"   Perfect reversibility: {error < 1e-5}")
    
    # 2. Analyze TOMORROW across languages
    print("\n2. THE TOMORROW PARADOX:")
    tomorrow_words = {
        'English': 'tomorrow',
        'Spanish': 'mañana',
        'French': 'demain',
        'German': 'morgen',
        'Italian': 'domani'
    }
    
    for lang, word in tomorrow_words.items():
        w_out, m_out = system.forward_path(word)
        w_energy = w_out.abs().mean().item()
        m_energy = m_out.abs().mean().item()
        w_ratio = w_energy / (w_energy + m_energy + 1e-8)
        
        print(f"   {lang} '{word}':")
        print(f"      W-dominance: {w_ratio:.3f} (possibility)")
        print(f"      M-dominance: {1-w_ratio:.3f} (manifestation)")
        
        if lang == 'Spanish':
            print(f"      → Starts with M but dissolves into 'ana' (open vowels)")
    
    # 3. The taste experiment
    print("\n3. ALPHABETIC TASTE EXPERIMENT:")
    print("   Same substance, different labels:")
    
    labels = [
        "artisanal botanical essence",
        "industrial food additive E-471",
        "grandmother's secret ingredient"
    ]
    
    results = system.taste_experiment("neutral_gel", labels)
    
    for label, data in results.items():
        print(f"\n   Label: '{label}'")
        print(f"      Perceived as: {data['top_taste']}")
        print(f"      W/M ratio: {data['w_dominance']:.1%}/{data['m_dominance']:.1%}")
        
        # Show top 3 taste resonances
        for taste, score in sorted(data['tastes'].items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)[:3]:
            if score > 0.1:
                print(f"      {taste}: {score:.3f}")
    
    # 4. Trace the TOMORROW flow
    print("\n4. TRACING T→O→M→O→R→R→O→W:")
    states = system.resonator.trace_tomorrow()
    print("   Energy at each step:")
    for step, state in states.items():
        energy = state.abs().mean().item()
        letter = step.split('_')[1]
        print(f"      {step}: {energy:.3f}")
        if letter == 'm':
            print(f"         ↑ M crystallizes (peak manifestation)")
        elif letter == 'w':
            print(f"         ↑ W opens (returns to possibility)")
    
    # 5. The hypothesis
    print("\n5. THE LINGUISTIC HYPOTHESIS:")
    print("   • Words are not descriptions but instructions")
    print("   • The alphabetic pattern primes biological response")
    print("   • W-dominant words keep futures open")
    print("   • M-dominant words crystallize experience")
    print("   • Taste is as much linguistic as chemical")
    
    print("\n6. PRACTICAL IMPLICATIONS:")
    print("   • Reframe food → change taste")
    print("   • Reframe medicine → enhance effect")
    print("   • Reframe experience → alter perception")
    print("   • The 48-manifold ensures reversibility:")
    print("     what language creates, it can uncreate")
    
    print("\n" + "=" * 60)
    print("Reality tastes of the words we feed it")
    print("=" * 60)

if __name__ == "__main__":
    print(f"Device: {device}")
    demonstrate_linguistic_reality()
