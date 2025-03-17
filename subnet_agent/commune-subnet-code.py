import os
import json
import logging
import asyncio
import time
import random
import hashlib
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from dataclasses import dataclass, field

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("COMpySubnet")

class ModelType(Enum):
    """Enum representing different types of inference models."""
    DIFFUSION = "diffusion"
    CHAT = "chat"
    IMAGE_CLASSIFICATION = "image_classification"
    AUDIO_PROCESSING = "audio_processing"
    EMBEDDING = "embedding"
    VIDEO_GENERATION = "video_generation"
    TEXT_TO_SPEECH = "text_to_speech"
    UNKNOWN = "unknown"


@dataclass
class ZKProof:
    """Represents a zero-knowledge proof for computational integrity."""
    circuit_id: str  # Unique identifier for the ZK circuit
    proof_data: bytes  # The actual proof data
    public_inputs: Dict[str, Any]  # Public inputs to the proof
    timestamp: float = field(default_factory=time.time)
    
    def verify(self) -> bool:
        """Verify the ZK proof (simplified implementation)."""
        # In a real implementation, this would use a ZKP verification library
        # For now, we'll simulate verification with a basic check
        return len(self.proof_data) > 0


@dataclass
class CircuitCommitment:
    """Represents a commitment to a ZK circuit design."""
    circuit_id: str
    circuit_hash: str  # Hash of the circuit implementation
    description: str
    owner_id: str
    creation_time: float = field(default_factory=time.time)
    
    @staticmethod
    def create_from_circuit(circuit_id: str, circuit_code: str, owner_id: str, description: str) -> 'CircuitCommitment':
        """Create a commitment from circuit code."""
        circuit_hash = hashlib.sha256(circuit_code.encode()).hexdigest()
        return CircuitCommitment(
            circuit_id=circuit_id,
            circuit_hash=circuit_hash,
            description=description,
            owner_id=owner_id
        )


@dataclass
class MinerInfo:
    """Information about a miner node."""
    id: str
    namespace: str
    model_type: ModelType
    endpoint: str
    capabilities: Dict = field(default_factory=dict)
    performance_metrics: Dict = field(default_factory=dict)
    is_active: bool = True
    last_validated: float = 0.0
    circuit_commitment: Optional[CircuitCommitment] = None
    is_source_of_truth: bool = False
    flagged: bool = False
    flag_reason: Optional[str] = None


@dataclass
class ValidatorInfo:
    """Information about a validator node."""
    id: str
    namespace: str
    specialization: ModelType
    endpoint: str
    capacity: int = 10  # Maximum number of miners this validator can handle
    current_load: int = 0
    reliability: float = 1.0  # 0.0 to 1.0 score
    is_active: bool = True
    circuit_commitment: Optional[CircuitCommitment] = None
    is_source_of_truth: bool = False
    flagged: bool = False
    flag_reason: Optional[str] = None


@dataclass
class DeterministicTestCase:
    """Known-output test case for ZKP circuit validation."""
    test_id: str
    model_type: ModelType
    inputs: Dict[str, Any]
    expected_outputs: Dict[str, Any]
    tolerance: float = 0.05  # Acceptable deviation for numerical outputs
    
    def check_result(self, actual_outputs: Dict[str, Any]) -> Tuple[bool, float, Optional[str]]:
        """Check if actual outputs match expected outputs within tolerance."""
        if set(self.expected_outputs.keys()) != set(actual_outputs.keys()):
            return False, 0.0, "Output keys don't match expected keys"
        
        # For simplicity, we'll just check numerical values and strings
        score = 0.0
        reason = None
        
        for key, expected in self.expected_outputs.items():
            actual = actual_outputs.get(key)
            
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                # Numerical comparison with tolerance
                if abs(expected - actual) > self.tolerance * abs(expected) and abs(expected) > 1e-10:
                    score = abs(expected - actual) / abs(expected)
                    reason = f"Numerical value for {key} outside tolerance"
                    return False, score, reason
            elif expected != actual:
                reason = f"Value mismatch for {key}"
                return False, 0.0, reason
            
        return True, 1.0, None


class TokenBurningMechanism:
    """Manages the token burning process for creating scarcity."""
    
    def __init__(self, initial_supply: int = 1_000_000_000):
        self.initial_supply = initial_supply
        self.current_supply = initial_supply
        self.burn_history: List[Dict] = []
    
    def burn_tokens(self, amount: int, reason: str, user_id: str = None) -> bool:
        """Burn tokens and record the transaction."""
        if amount <= 0:
            return False
        
        if amount > self.current_supply:
            logger.warning(f"Attempted to burn {amount} tokens but only {self.current_supply} remain")
            return False
        
        self.current_supply -= amount
        
        burn_record = {
            "amount": amount,
            "timestamp": time.time(),
            "reason": reason,
            "user_id": user_id,
            "remaining_supply": self.current_supply
        }
        
        self.burn_history.append(burn_record)
        logger.info(f"Burned {amount} tokens. Reason: {reason}. Remaining supply: {self.current_supply}")
        
        return True
    
    def calculate_module_burn_rate(self, module_quality: float, usage_intensity: float) -> int:
        """Calculate how many tokens to burn for module usage."""
        # Higher quality modules and more intensive usage burn more tokens
        base_rate = 10  # Base token burn rate
        quality_multiplier = 1.0 + (module_quality * 2.0)  # Quality from 0-1, multiplier from 1-3
        usage_multiplier = usage_intensity  # Usage intensity from 0-1
        
        burn_amount = int(base_rate * quality_multiplier * usage_multiplier)
        return max(1, burn_amount)  # Ensure at least 1 token is burned
    
    def get_burn_statistics(self) -> Dict:
        """Get statistics about token burning."""
        if not self.burn_history:
            return {
                "total_burned": 0,
                "percent_burned": 0,
                "burn_rate_per_day": 0
            }
        
        total_burned = self.initial_supply - self.current_supply
        percent_burned = (total_burned / self.initial_supply) * 100
        
        # Calculate burn rate per day over the last 30 days
        thirty_days_ago = time.time() - (30 * 24 * 60 * 60)
        recent_burns = [b for b in self.burn_history if b["timestamp"] > thirty_days_ago]
        
        if recent_burns:
            recent_total = sum(b["amount"] for b in recent_burns)
            burn_rate_per_day = recent_total / 30
        else:
            burn_rate_per_day = 0
        
        return {
            "total_burned": total_burned,
            "percent_burned": percent_burned,
            "burn_rate_per_day": burn_rate_per_day,
            "current_supply": self.current_supply
        }


class NamespaceRegistry:
    """Registry for mapping namespaces to model types."""
    
    def __init__(self):
        self.namespace_mappings = {
            "stable-diffusion": ModelType.DIFFUSION,
            "sd": ModelType.DIFFUSION,
            "midjourney": ModelType.DIFFUSION,
            "dalle": ModelType.DIFFUSION,
            
            "gpt": ModelType.CHAT,
            "llama": ModelType.CHAT,
            "claude": ModelType.CHAT,
            "mistral": ModelType.CHAT,
            "chat": ModelType.CHAT,
            
            "vit": ModelType.IMAGE_CLASSIFICATION,
            "resnet": ModelType.IMAGE_CLASSIFICATION,
            "yolo": ModelType.IMAGE_CLASSIFICATION,
            "image-class": ModelType.IMAGE_CLASSIFICATION,
            
            "whisper": ModelType.AUDIO_PROCESSING,
            "audio": ModelType.AUDIO_PROCESSING,
            "wav2vec": ModelType.AUDIO_PROCESSING,
            
            "embed": ModelType.EMBEDDING,
            "embedding": ModelType.EMBEDDING,
            "sentence-transformer": ModelType.EMBEDDING,
            
            "video": ModelType.VIDEO_GENERATION,
            "sora": ModelType.VIDEO_GENERATION,
            "gen-video": ModelType.VIDEO_GENERATION,
            
            "tts": ModelType.TEXT_TO_SPEECH,
            "eleven": ModelType.TEXT_TO_SPEECH,
            "speech": ModelType.TEXT_TO_SPEECH
        }
        
    def get_model_type(self, namespace: str) -> ModelType:
        """Determine the model type from a namespace."""
        # Check for exact matches
        if namespace in self.namespace_mappings:
            return self.namespace_mappings[namespace]
        
        # Check for partial matches in the namespace
        for key, model_type in self.namespace_mappings.items():
            if key in namespace.lower():
                return model_type
        
        # If no match is found
        return ModelType.UNKNOWN
    
    def register_namespace(self, namespace: str, model_type: ModelType) -> None:
        """Register a new namespace mapping."""
        self.namespace_mappings[namespace] = model_type
        logger.info(f"Registered new namespace mapping: {namespace} -> {model_type.value}")


class CircuitCommitmentRepository:
    """Stores and manages ZKP circuit commitments."""
    
    def __init__(self):
        self.commitments: Dict[str, CircuitCommitment] = {}
    
    def register_commitment(self, commitment: CircuitCommitment) -> bool:
        """Register a new circuit commitment."""
        if commitment.circuit_id in self.commitments:
            existing = self.commitments[commitment.circuit_id]
            if existing.circuit_hash != commitment.circuit_hash:
                logger.warning(f"Circuit ID collision with different hash: {commitment.circuit_id}")
                return False
            # Same hash, so it's effectively the same commitment
            return True
        
        self.commitments[commitment.circuit_id] = commitment
        logger.info(f"Registered new circuit commitment: {commitment.circuit_id}")
        return True
    
    def verify_commitment(self, circuit_id: str, circuit_hash: str) -> bool:
        """Verify that a circuit matches its registered commitment."""
        if circuit_id not in self.commitments:
            logger.warning(f"No commitment found for circuit ID: {circuit_id}")
            return False
        
        return self.commitments[circuit_id].circuit_hash == circuit_hash


class DeterministicTestRepository:
    """Repository of deterministic test cases for verification."""
    
    def __init__(self):
        self.test_cases: Dict[ModelType, List[DeterministicTestCase]] = {
            model_type: [] for model_type in ModelType
        }
    
    def add_test_case(self, test_case: DeterministicTestCase) -> None:
        """Add a test case to the repository."""
        self.test_cases[test_case.model_type].append(test_case)
        logger.info(f"Added test case {test_case.test_id} for {test_case.model_type.value}")
    
    def get_test_cases(self, model_type: ModelType, count: int = 3) -> List[DeterministicTestCase]:
        """Get a number of test cases for a specific model type."""
        available = self.test_cases[model_type]
        if not available:
            logger.warning(f"No test cases available for {model_type.value}")
            return []
        
        # Return randomly selected test cases, up to the requested count
        if len(available) <= count:
            return available
        
        return random.sample(available, count)


class TrustVerificationSystem:
    """System for verifying trust in miners and validators."""
    
    def __init__(self, testing_frequency: float = 0.1):
        self.testing_frequency = testing_frequency  # Probability of triggering a test
        self.test_repository = DeterministicTestRepository()
        
    async def verify_miner(self, miner: MinerInfo, validators: List[ValidatorInfo]) -> Tuple[bool, Optional[str]]:
        """Verify a miner by cross-checking with multiple validators."""
        # Only test with some probability, unless the miner is already flagged
        if not miner.flagged and random.random() > self.testing_frequency:
            return True, None
        
        # Get validators for this model type, excluding the flagged ones
        matching_validators = [
            v for v in validators 
            if v.specialization == miner.model_type and v.is_active and not v.flagged
        ]
        
        if len(matching_validators) < 2:
            logger.warning(f"Not enough validators to cross-check miner {miner.id}")
            return True, None  # Can't verify without enough validators
        
        # Select two different validators for cross-checking
        test_validators = random.sample(matching_validators, 2)
        
        # Get test cases for this model type
        test_cases = self.test_repository.get_test_cases(miner.model_type, count=2)
        if not test_cases:
            logger.warning(f"No test cases available for {miner.model_type.value}")
            return True, None
        
        # Run tests with each validator and compare results
        results = []
        
        for validator in test_validators:
            for test_case in test_cases:
                # In a real implementation, this would call the validator's API
                # Here we simulate the validation
                await asyncio.sleep(0.2)  # Simulate network delay
                
                # Simulate validation result (could fail for flagged miners)
                if miner.flagged and random.random() < 0.8:  # Higher chance of failure for flagged miners
                    success = False
                    score = random.uniform(0.2, 0.5)
                    reason = "Output quality below threshold"
                else:
                    success = random.random() < 0.9  # 90% success rate for normal miners
                    score = random.uniform(0.7, 1.0) if success else random.uniform(0.3, 0.6)
                    reason = None if success else "Unexpected output format"
                
                results.append((validator.id, test_case.test_id, success, score, reason))
        
        # Analyze results to determine if there's a problem
        success_rate = sum(1 for _, _, success, _, _ in results if success) / len(results)
        
        if success_rate < 0.7:  # If less than 70% of tests pass
            reasons = [reason for _, _, success, _, reason in results if not success and reason]
            main_reason = max(set(reasons), key=reasons.count) if reasons else "Inconsistent results across validators"
            return False, main_reason
        
        return True, None
        
    async def verify_validator(self, validator: ValidatorInfo, truth_miners: List[MinerInfo]) -> Tuple[bool, Optional[str]]:
        """Verify a validator against source of truth miners."""
        # Only test with some probability, unless the validator is already flagged
        if not validator.flagged and random.random() > self.testing_frequency:
            return True, None
        
        # Get source of truth miners for this model type
        matching_truth_miners = [
            m for m in truth_miners 
            if m.model_type == validator.specialization and m.is_active and m.is_source_of_truth
        ]
        
        if not matching_truth_miners:
            logger.warning(f"No source of truth miners for {validator.specialization.value}")
            return True, None
        
        # Select a truth miner to use for testing
        truth_miner = random.choice(matching_truth_miners)
        
        # Get test cases
        test_cases = self.test_repository.get_test_cases(validator.specialization, count=3)
        if not test_cases:
            return True, None
        
        # Run tests and compare to expected outputs
        results = []
        
        for test_case in test_cases:
            # In a real implementation, this would call the validator's API with the truth miner
            # Here we simulate the validation
            await asyncio.sleep(0.2)  # Simulate network delay
            
            # Generate a simulated actual output that's close to expected
            # In reality, this would come from the validator scoring the truth miner
            actual_outputs = {}
            for key, expected in test_case.expected_outputs.items():
                if isinstance(expected, (int, float)):
                    # Add some noise, more for flagged validators
                    noise_level = 0.2 if validator.flagged else 0.05
                    actual_outputs[key] = expected * (1 + random.uniform(-noise_level, noise_level))
                else:
                    actual_outputs[key] = expected
            
            # Check if the result matches expectations
            success, score, reason = test_case.check_result(actual_outputs)
            results.append((test_case.test_id, success, score, reason))
        
        # Analyze results
        success_rate = sum(1 for _, success, _, _ in results if success) / len(results)
        
        if success_rate < 0.7:  # If less than 70% of tests pass
            reasons = [reason for _, _, _, reason in results if reason]
            main_reason = max(set(reasons), key=reasons.count) if reasons else "Inconsistent scoring against truth models"
            return False, main_reason
        
        return True, None


class ZKPVerificationSystem:
    """System for verifying ZKP circuit integrity."""
    
    def __init__(self):
        self.circuit_repository = CircuitCommitmentRepository()
    
    def register_circuit(self, commitment: CircuitCommitment) -> bool:
        """Register a new ZKP circuit commitment."""
        return self.circuit_repository.register_commitment(commitment)
    
    def verify_proof(self, proof: ZKProof) -> bool:
        """Verify a ZK proof."""
        # First check if the circuit is registered
        if proof.circuit_id not in self.circuit_repository.commitments:
            logger.warning(f"Unknown circuit ID in proof: {proof.circuit_id}")
            return False
        
        # Then verify the proof itself (using the simplified implementation)
        return proof.verify()
    
    def verify_circuit_integrity(self, circuit_id: str, circuit_code: str) -> bool:
        """Verify that a circuit implementation matches its commitment."""
        circuit_hash = hashlib.sha256(circuit_code.encode()).hexdigest()
        return self.circuit_repository.verify_commitment(circuit_id, circuit_hash)


class SubnetAgent:
    """Agent that matches miners to validators based on model type."""
    
    def __init__(self, zkp_system: ZKPVerificationSystem, trust_system: TrustVerificationSystem):
        self.namespace_registry = NamespaceRegistry()
        self.miners: Dict[str, MinerInfo] = {}
        self.validators: Dict[str, ValidatorInfo] = {}
        self.matches: Dict[str, str] = {}  # miner_id -> validator_id
        self.zkp_system = zkp_system
        self.trust_system = trust_system
        
    def register_miner(self, miner_info: MinerInfo) -> bool:
        """Register a new miner in the subnet."""
        # If model type is unknown, try to determine it from the namespace
        if miner_info.model_type == ModelType.UNKNOWN:
            miner_info.model_type = self.namespace_registry.get_model_type(miner_info.namespace)
        
        # Check if the miner has a valid circuit commitment if not a source of truth
        if not miner_info.is_source_of_truth and miner_info.circuit_commitment is None:
            logger.warning(f"Miner {miner_info.id} has no circuit commitment")
            return False
        
        self.miners[miner_info.id] = miner_info
        logger.info(f"Registered miner: {miner_info.id} with namespace {miner_info.namespace} as {miner_info.model_type.value}")
        
        # Try to match the miner immediately
        self._match_miner(miner_info.id)
        return True
    
    def register_validator(self, validator_info: ValidatorInfo) -> bool:
        """Register a new validator in the subnet."""
        # Check if the validator has a valid circuit commitment if not a source of truth
        if not validator_info.is_source_of_truth and validator_info.circuit_commitment is None:
            logger.warning(f"Validator {validator_info.id} has no circuit commitment")
            return False
            
        self.validators[validator_info.id] = validator_info
        logger.info(f"Registered validator: {validator_info.id} specializing in {validator_info.specialization.value}")
        
        # Check if any unmatched miners can now be matched
        self._rematch_miners()
        return True
    
    def _match_miner(self, miner_id: str) -> Optional[str]:
        """Match a miner to an appropriate validator."""
        if miner_id not in self.miners:
            logger.warning(f"Cannot match unknown miner: {miner_id}")
            return None
        
        miner = self.miners[miner_id]
        
        # Don't match flagged miners
        if miner.flagged:
            logger.warning(f"Cannot match flagged miner: {miner_id}")
            return None
        
        # Find validators specializing in this model type
        matching_validators = [
            v for v in self.validators.values() 
            if v.specialization == miner.model_type and v.is_active and v.current_load < v.capacity and not v.flagged
        ]
        
        if not matching_validators:
            logger.warning(f"No available validators for miner {miner_id} with type {miner.model_type.value}")
            return None
        
        # Sort by load and reliability
        matching_validators.sort(key=lambda v: (v.current_load, -v.reliability))
        
        # Match with the most available reliable validator
        best_validator = matching_validators[0]
        best_validator.current_load += 1
        
        # Record the match
        self.matches[miner_id] = best_validator.id
        logger.info(f"Matched miner {miner_id} to validator {best_validator.id}")
        
        return best_validator.id
    
    def _rematch_miners(self) -> None:
        """Attempt to rematch any unmatched miners."""
        unmatched_miners = [m.id for m in self.miners.values() if m.id not in self.matches and m.is_active and not m.flagged]
        
        for miner_id in unmatched_miners:
            self._match_miner(miner_id)
    
    def get_validator_for_miner(self, miner_id: str) -> Optional[ValidatorInfo]:
        """Get the validator assigned to a miner."""
        if miner_id not in self.matches:
            return None
        
        validator_id = self.matches[miner_id]
        return self.validators.get(validator_id)
    
    def update_miner_status(self, miner_id: str, is_active: bool) -> bool:
        """Update a miner's active status."""
        if miner_id not in self.miners:
            return False
        
        self.miners[miner_id].is_active = is_active
        
        # If deactivated, remove from matches and update validator load
        if not is_active and miner_id in self.matches:
            validator_id = self.matches[miner_id]
            if validator_id in self.validators:
                self.validators[validator_id].current_load -= 1
            del self.matches[miner_id]
            logger.info(f"Miner {miner_id} deactivated and unmatched")
        
        # If activated, try to match
        elif is_active and miner_id not in self.matches and not self.miners[miner_id].flagged:
            self._match_miner(miner_id)
        
        return True
    
    def update_validator_status(self, validator_id: str, is_active: bool) -> bool:
        """Update a validator's active status."""
        if validator_id not in self.validators:
            return False
        
        prev_status = self.validators[validator_id].is_active
        self.validators[validator_id].is_active = is_active
        
        # If deactivated, remove all matches to this validator
        if not is_active and prev_status:
            affected_miners = [m_id for m_id, v_id in self.matches.items() if v_id == validator_id]
            for miner_id in affected_miners:
                del self.matches[miner_id]
            
            # Reset load
            self.validators[validator_id].current_load = 0
            logger.info(f"Validator {validator_id} deactivated, unmatched {len(affected_miners)} miners")
            
            # Try to rematch affected miners
            for miner_id in affected_miners:
                self._match_miner(miner_id)
        
        # If activated, check for potential matches
        elif is_active and not prev_status:
            self._rematch_miners()
        
        return True
    
    async def perform_trust_verification(self) -> Tuple[List[str], List[str]]:
        """Perform trust verification on a sample of miners and validators."""
        # Get source of truth miners for validator verification
        truth_miners = [m for m in self.miners.values() if m.is_source_of_truth and m.is_active]
        
        # Get validators for miner cross-verification
        active_validators = [v for v in self.validators.values() if v.is_active]
        
        # Select a sample of miners and validators to test
        miners_to_test = random.sample(list(self.miners.values()), min(5, len(self.miners))) if self.miners else []
        validators_to_test = random.sample(list(self.validators.values()), min(3, len(self.validators))) if self.validators else []
        
        # Track which ones get flagged
        flagged_miners = []
        flagged_validators = []
        
        # Test miners
        for miner in miners_to_test:
            trusted, reason = await self.trust_system.verify_miner(miner, active_validators)
            if not trusted:
                miner.flagged = True
                miner.flag_reason = reason
                logger.warning(f"Miner {miner.id} flagged: {reason}")
                flagged_miners.append(miner.id)
                
                # Remove from matches if flagged
                if miner.id in self.matches:
                    validator_id = self.matches[miner.id]
                    if validator_id in self.validators:
                        self.validators[validator_id].current_load -= 1
                    del self.matches[miner.id]
                    logger.info(f"Flagged miner {miner.id} removed from matches")
        
        # Test validators
        for validator in validators_to_test:
            trusted, reason = await self.trust_system.verify_validator(validator, truth_miners)
            if not trusted:
                validator.flagged = True
                validator.flag_reason = reason
                logger.warning(f"Validator {validator.id} flagged: {reason}")
                flagged_validators.append(validator.id)
                
                # Remove matches to flagged validator
                affected_miners = [m_id for m_id, v_id in self.matches.items() if v_id == validator.id]
                for miner_id in affected_miners:
                    del self.matches[miner_id]
                
                # Reset load
                validator.current_load = 0
                logger.info(f"Flagged validator {validator.id}, unmatched {len(affected_miners)} miners")
                
                # Try to rematch affected miners
                for miner_id in affected_miners:
                    self._match_miner(miner_id)
        
        return flagged_miners, flagged_validators


class ValidationResult:
    """Stores the result of a validation operation."""
    
    def __init__(self, miner_id: str, validator_id: str, score: float, details: Dict = None):
        self.miner_id = miner_id
        self.validator_id = validator_id
        self.score = score  # 0.0 to 1.0
        self.details = details or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        """Convert the result to a dictionary."""
        return {
            "miner_id": self.miner_id,
            "validator_id": self.validator_id,
            "score": self.score,
            "details": self.details,
            "timestamp": self.timestamp
        }


@dataclass
class ModuleInfo:
    """Information about a validated module in the registry."""
    id: str
    name: str
    namespace: str
    model_type: ModelType
    version: str
    miner_id: str
    description: str
    api_endpoint: str
    average_score: float
    usage_count: int = 0
    creation_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    token_burn_rate: int = 10  # Base tokens to burn per use
    parameters: Dict[str, Any] = field(default_factory=dict)
    example_usages: List[Dict] = field(default_factory=list)


class ModuleRegistry:
    """Registry of validated modules available for use."""
    
    def __init__(self, token_burning: TokenBurningMechanism):
        self.modules: Dict[str, ModuleInfo] = {}
        self.token_burning = token_burning
    
    def register_module(self, module: ModuleInfo) -> bool:
        """Register a module in the registry."""
        if module.id in self.modules:
            # Update existing module
            existing = self.modules[module.id]
            existing.version = module.version
            existing.description = module.description
            existing.api_endpoint = module.api_endpoint
            existing.average_score = module.average_score
            existing.last_updated = time.time()
            existing.parameters = module.parameters
            existing.example_usages = module.example_usages
            
            logger.info(f"Updated module {module.id} to version {module.version}")
            return True
        
        # Add new module
        self.modules[module.id] = module
        logger.info(f"Registered new module: {module.id} - {module.name}")
        return True
    
    def get_module(self, module_id: str) -> Optional[ModuleInfo]:
        """Get a module by ID."""
        return self.modules.get(module_id)
    
    def search_modules(self, model_type: Optional[ModelType] = None, query: str = "") -> List[ModuleInfo]:
        """Search for modules by type and query string."""
        results = []
        
        for module in self.modules.values():
            # Filter by model type if specified
            if model_type and module.model_type != model_type:
                continue
            
            # Filter by query if specified
            if query and query.lower() not in module.name.lower() and query.lower() not in module.description.lower():
                continue
            
            results.append(module)
        
        # Sort by score and then usage count
        results.sort(key=lambda m: (-m.average_score, -m.usage_count))
        
        return results
    
    def use_module(self, module_id: str, usage_intensity: float = 1.0, user_id: Optional[str] = None) -> bool:
        """Record usage of a module and burn tokens accordingly."""
        if module_id not in self.modules:
            return False
        
        module = self.modules[module_id]
        module.usage_count += 1
        
        # Calculate token burn amount based on module quality and usage intensity
        burn_amount = self.token_burning.calculate_module_burn_rate(module.average_score, usage_intensity)
        
        # Burn tokens
        reason = f"Module usage: {module.name} (ID: {module_id})"
        success = self.token_burning.burn_tokens(burn_amount, reason, user_id)
        
        if success:
            logger.info(f"Burned {burn_amount} tokens for module {module_id} usage")
        
        return success
    
    def update_module_score(self, module_id: str, new_score: float) -> bool:
        """Update a module's score based on new validation results."""
        if module_id not in self.modules:
            return False
        
        module = self.modules[module_id]
        
        # Update using a weighted average that gives more weight to recent scores
        alpha = 0.3  # Weight for new score (0-1)
        module.average_score = (alpha * new_score) + ((1 - alpha) * module.average_score)
        module.last_updated = time.time()
        
        logger.info(f"Updated module {module_id} score to {module.average_score:.2f}")
        return True


class COMpySubnet:
    """Main subnet class that orchestrates miners, validators, and the agent."""
    
    def __init__(self, subnet_id: str):
        self.subnet_id = subnet_id
        self.zkp_system = ZKPVerificationSystem()
        self.trust_system = TrustVerificationSystem()
        self.agent = SubnetAgent(self.zkp_system, self.trust_system)
        self.token_burning = TokenBurningMechanism()
        self.module_registry = ModuleRegistry(self.token_burning)
        self.validation_results: List[ValidationResult] = []
        
        logger.info(f"Initialized COMpySubnet with ID: {subnet_id}")
        
        # Initialize test cases
        self._initialize_test_cases()
    
    def _initialize_test_cases(self):
        """Initialize some default test cases for different model types."""
        # Add test cases for diffusion models
        diffusion_test = DeterministicTestCase(
            test_id="diffusion_test_1",
            model_type=ModelType.DIFFUSION,
            inputs={"prompt": "a photo of a cat", "steps": 50},
            expected_outputs={"image_hash": "dummy_hash_value", "quality_score": 0.85}
        )
        self.trust_system.test_repository.add_test_case(diffusion_test)
        
        # Add test cases for chat models
        chat_test = DeterministicTestCase(
            test_id="chat_test_1",
            model_type=ModelType.CHAT,
            inputs={"prompt": "What is the capital of France?"},
            expected_outputs={"contains_paris": 1.0, "word_count": 25}
        )
        self.trust_system.test_repository.add_test_case(chat_test)
        
        # Add more test cases as needed for other model types
    
    def register_miner(self, id: str, namespace: str, endpoint: str, 
                       circuit_code: str = None, is_source_of_truth: bool = False,
                       capabilities: Dict = None) -> bool:
        """Register a miner with the subnet."""
        # Determine model type from namespace
        model_type = self.agent.namespace_registry.get_model_type(namespace)
        
        # Create circuit commitment if provided and not a source of truth
        circuit_commitment = None
        if not is_source_of_truth and circuit_code:
            circuit_id = f"{id}_circuit"
            description = f"ZKP circuit for {namespace} miner"
            circuit_commitment = CircuitCommitment.create_from_circuit(
                circuit_id=circuit_id,
                circuit_code=circuit_code,
                owner_id=id,
                description=description
            )
            
            # Register the circuit commitment
            self.zkp_system.register_circuit(circuit_commitment)
        
        miner_info = MinerInfo(
            id=id,
            namespace=namespace,
            model_type=model_type,
            endpoint=endpoint,
            capabilities=capabilities or {},
            circuit_commitment=circuit_commitment,
            is_source_of_truth=is_source_of_truth
        )
        
        return self.agent.register_miner(miner_info)
    
    def register_validator(self, id: str, namespace: str, specialization: str, 
                           endpoint: str, circuit_code: str = None, 
                           is_source_of_truth: bool = False, capacity: int = 10) -> bool:
        """Register a validator with the subnet."""
        try:
            model_type = ModelType(specialization)
        except ValueError:
            # Try to determine from namespace if direct mapping fails
            model_type = self.agent.namespace_registry.get_model_type(specialization)
        
        # Create circuit commitment if provided and not a source of truth
        circuit_commitment = None
        if not is_source_of_truth and circuit_code:
            circuit_id = f"{id}_circuit"
            description = f"ZKP circuit for {namespace} validator"
            circuit_commitment = CircuitCommitment.create_from_circuit(
                circuit_id=circuit_id,
                circuit_code=circuit_code,
                owner_id=id,
                description=description
            )
            
            # Register the circuit commitment
            self.zkp_system.register_circuit(circuit_commitment)
        
        validator_info = ValidatorInfo(
            id=id,
            namespace=namespace,
            specialization=model_type,
            endpoint=endpoint,
            capacity=capacity,
            circuit_commitment=circuit_commitment,
            is_source_of_truth=is_source_of_truth
        )
        
        return self.agent.register_validator(validator_info)
    
    async def trigger_validation(self, miner_id: str) -> Optional[ValidationResult]:
        """Trigger validation for a miner."""
        validator_info = self.agent.get_validator_for_miner(miner_id)
        if not validator_info:
            logger.warning(f"No validator found for miner {miner_id}")
            return None
        
        miner_info = self.agent.miners.get(miner_id)
        if not miner_info:
            logger.warning(f"Miner {miner_id} not found")
            return None
        
        logger.info(f"Triggering validation for miner {miner_id} by validator {validator_info.id}")
        
        # In a real implementation, you would call the validator's endpoint
        # Here we simulate the validation process
        await asyncio.sleep(1)  # Simulate network delay
        
        # Mock validation result
        score = random.uniform(0.75, 0.95)  # This would come from the validator
        result = ValidationResult(
            miner_id=miner_id,
            validator_id=validator_info.id,
            score=score,
            details={"latency": random.randint(100, 200), "quality": random.uniform(0.7, 0.95), "availability": random.uniform(0.8, 1.0)}
        )
        
        # Update miner's last validated timestamp
        miner_info.last_validated = result.timestamp
        
        # Store the result
        self.validation_results.append(result)
        
        # Create or update module in registry if score is good enough
        if score >= 0.7:
            module_id = f"{miner_id}_module"
            module = ModuleInfo(
                id=module_id,
                name=f"{miner_info.namespace} Module",
                namespace=miner_info.namespace,
                model_type=miner_info.model_type,
                version="1.0",
                miner_id=miner_id,
                description=f"Module for {miner_info.namespace} inference",
                api_endpoint=miner_info.endpoint,
                average_score=score,
                token_burn_rate=int(score * 20)  # Higher scores mean higher burn rates
            )
            self.module_registry.register_module(module)
        
        return result
    
    def get_emission_allocation(self, time_period: Tuple[float, float] = None) -> Dict[str, float]:
        """Calculate emission allocations based on validation results."""
        # Filter results by time period if specified
        results = self.validation_results
        if time_period:
            start_time, end_time = time_period
            results = [r for r in results if start_time <= r.timestamp <= end_time]
        
        if not results:
            return {}
        
        # Group by miner
        miner_scores = {}
        for result in results:
            if result.miner_id not in miner_scores:
                miner_scores[result.miner_id] = []
            miner_scores[result.miner_id].append(result.score)
        
        # Calculate average score per miner
        avg_scores = {
            miner_id: sum(scores) / len(scores) 
            for miner_id, scores in miner_scores.items()
        }
        
        # Normalize to get allocation percentages
        total_score = sum(avg_scores.values())
        if total_score == 0:
            return {miner_id: 0 for miner_id in avg_scores}
        
        allocations = {
            miner_id: score / total_score 
            for miner_id, score in avg_scores.items()
        }
        
        return allocations
    
    async def run_trust_verification(self) -> Tuple[List[str], List[str]]:
        """Run a round of trust verification."""
        return await self.agent.perform_trust_verification()
    
    def get_token_economy_status(self) -> Dict:
        """Get current status of the token economy."""
        return self.token_burning.get_burn_statistics()
    
    def get_subnet_status(self) -> Dict:
        """Get the current status of the subnet."""
        return {
            "subnet_id": self.subnet_id,
            "miners": len(self.agent.miners),
            "validators": len(self.agent.validators),
            "active_miners": sum(1 for m in self.agent.miners.values() if m.is_active),
            "active_validators": sum(1 for v in self.agent.validators.values() if v.is_active),
            "flagged_miners": sum(1 for m in self.agent.miners.values() if m.flagged),
            "flagged_validators": sum(1 for v in self.agent.validators.values() if v.flagged),
            "matches": len(self.agent.matches),
            "validation_results": len(self.validation_results),
            "registered_modules": len(self.module_registry.modules),
            "token_economy": self.get_token_economy_status()
        }


# Example usage
async def main():
    # Create a subnet
    subnet = COMpySubnet("compy-inference-subnet")
    
    # Register source of truth validators
    subnet.register_validator(
        id="truth-val-001", 
        namespace="diffusion-truth-verification", 
        specialization="diffusion", 
        endpoint="http://truth-validator1:8000",
        is_source_of_truth=True
    )
    
    subnet.register_validator(
        id="truth-val-002", 
        namespace="chat-truth-verification", 
        specialization="chat", 
        endpoint="http://truth-validator2:8000",
        is_source_of_truth=True
    )
    
    # Register regular validators with ZKP circuits
    subnet.register_validator(
        id="val-001", 
        namespace="diffusion-verification", 
        specialization="diffusion", 
        endpoint="http://validator1:8000",
        circuit_code="function verifyDiffusion(inputs, outputs) { /* ZKP circuit code */ }"
    )
    
    subnet.register_validator(
        id="val-002", 
        namespace="chat-verification", 
        specialization="chat", 
        endpoint="http://validator2:8000",
        circuit_code="function verifyChat(inputs, outputs) { /* ZKP circuit code */ }"
    )
    
    # Register source of truth miners
    subnet.register_miner(
        id="truth-miner-001", 
        namespace="stable-diffusion-truth", 
        endpoint="http://truth-miner1:7000",
        is_source_of_truth=True
    )
    
    subnet.register_miner(
        id="truth-miner-002", 
        namespace="gpt-truth", 
        endpoint="http://truth-miner2:7000",
        is_source_of_truth=True
    )
    
    # Register regular miners with ZKP circuits
    subnet.register_miner(
        id="miner-001", 
        namespace="stable-diffusion-xl", 
        endpoint="http://miner1:7000",
        circuit_code="function generateDiffusion(prompt, params) { /* ZKP circuit code */ }"
    )
    
    subnet.register_miner(
        id="miner-002", 
        namespace="gpt-4-turbo", 
        endpoint="http://miner2:7000", 
        circuit_code="function generateChat(prompt, params) { /* ZKP circuit code */ }"
    )
    
    # Print current matches
    for miner_id, validator_id in subnet.agent.matches.items():
        miner = subnet.agent.miners[miner_id]
        validator = subnet.agent.validators[validator_id]
        print(f"Miner {miner_id} ({miner.model_type.value}) matched with Validator {validator_id}")
    
    # Trigger validation for miners
    for miner_id in ["miner-001", "miner-002"]:
        result = await subnet.trigger_validation(miner_id)
        if result:
            print(f"Validation result for {miner_id}: {result.score:.2f}")
    
    # Run trust verification
    flagged_miners, flagged_validators = await subnet.run_trust_verification()
    if flagged_miners:
        print(f"Flagged miners: {flagged_miners}")
    if flagged_validators:
        print(f"Flagged validators: {flagged_validators}")
    
    # Simulate module usage by end users
    for _ in range(10):
        modules = subnet.module_registry.search_modules()
        if modules:
            module = modules[0]  # Use the highest-ranked module
            subnet.module_registry.use_module(module.id, usage_intensity=random.uniform(0.5, 1.0))
    
    # Get emission allocations
    allocations = subnet.get_emission_allocation()
    print("Emission allocations:", allocations)
    
    # Get subnet status
    status = subnet.get_subnet_status()
    print("Subnet status:", status)
    
    # Get token economy status
    token_status = subnet.get_token_economy_status()
    print("Token economy status:", token_status)


if __name__ == "__main__":
    asyncio.run(main())
