import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import asyncio
import time
from dataclasses import dataclass, field

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CommuneSubnet")

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


class SubnetAgent:
    """Agent that matches miners to validators based on model type."""
    
    def __init__(self):
        self.namespace_registry = NamespaceRegistry()
        self.miners: Dict[str, MinerInfo] = {}
        self.validators: Dict[str, ValidatorInfo] = {}
        self.matches: Dict[str, str] = {}  # miner_id -> validator_id
        
    def register_miner(self, miner_info: MinerInfo) -> bool:
        """Register a new miner in the subnet."""
        # If model type is unknown, try to determine it from the namespace
        if miner_info.model_type == ModelType.UNKNOWN:
            miner_info.model_type = self.namespace_registry.get_model_type(miner_info.namespace)
        
        self.miners[miner_info.id] = miner_info
        logger.info(f"Registered miner: {miner_info.id} with namespace {miner_info.namespace} as {miner_info.model_type.value}")
        
        # Try to match the miner immediately
        self._match_miner(miner_info.id)
        return True
    
    def register_validator(self, validator_info: ValidatorInfo) -> bool:
        """Register a new validator in the subnet."""
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
        
        # Find validators specializing in this model type
        matching_validators = [
            v for v in self.validators.values() 
            if v.specialization == miner.model_type and v.is_active and v.current_load < v.capacity
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
        unmatched_miners = [m.id for m in self.miners.values() if m.id not in self.matches and m.is_active]
        
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
        elif is_active and miner_id not in self.matches:
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


class CommuneSubnet:
    """Main subnet class that orchestrates miners, validators, and the agent."""
    
    def __init__(self, subnet_id: str):
        self.subnet_id = subnet_id
        self.agent = SubnetAgent()
        self.validation_results: List[ValidationResult] = []
        logger.info(f"Initialized CommuneSubnet with ID: {subnet_id}")
    
    def register_miner(self, id: str, namespace: str, endpoint: str, 
                       capabilities: Dict = None) -> bool:
        """Register a miner with the subnet."""
        # Determine model type from namespace
        model_type = self.agent.namespace_registry.get_model_type(namespace)
        
        miner_info = MinerInfo(
            id=id,
            namespace=namespace,
            model_type=model_type,
            endpoint=endpoint,
            capabilities=capabilities or {}
        )
        
        return self.agent.register_miner(miner_info)
    
    def register_validator(self, id: str, namespace: str, specialization: str, 
                          endpoint: str, capacity: int = 10) -> bool:
        """Register a validator with the subnet."""
        try:
            model_type = ModelType(specialization)
        except ValueError:
            # Try to determine from namespace if direct mapping fails
            model_type = self.agent.namespace_registry.get_model_type(specialization)
        
        validator_info = ValidatorInfo(
            id=id,
            namespace=namespace,
            specialization=model_type,
            endpoint=endpoint,
            capacity=capacity
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
        score = 0.85  # This would come from the validator
        result = ValidationResult(
            miner_id=miner_id,
            validator_id=validator_info.id,
            score=score,
            details={"latency": 120, "quality": 0.88, "availability": 0.82}
        )
        
        # Update miner's last validated timestamp
        miner_info.last_validated = result.timestamp
        
        # Store the result
        self.validation_results.append(result)
        
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
    
    def get_subnet_status(self) -> Dict:
        """Get the current status of the subnet."""
        return {
            "subnet_id": self.subnet_id,
            "miners": len(self.agent.miners),
            "validators": len(self.agent.validators),
            "active_miners": sum(1 for m in self.agent.miners.values() if m.is_active),
            "active_validators": sum(1 for v in self.agent.validators.values() if v.is_active),
            "matches": len(self.agent.matches),
            "validation_results": len(self.validation_results)
        }


# Example usage
async def main():
    # Create a subnet for image processing
    subnet = CommuneSubnet("image-processing-subnet")
    
    # Register validators
    subnet.register_validator("val-001", "diffusion-verification", "diffusion", "http://validator1:8000")
    subnet.register_validator("val-002", "chat-verification", "chat", "http://validator2:8000")
    subnet.register_validator("val-003", "image-class-verification", "image_classification", "http://validator3:8000")
    
    # Register miners
    subnet.register_miner("miner-001", "stable-diffusion-xl", "http://miner1:7000")
    subnet.register_miner("miner-002", "gpt-4-turbo", "http://miner2:7000") 
    subnet.register_miner("miner-003", "llama-3-70b", "http://miner3:7000")
    subnet.register_miner("miner-004", "sd-turbo", "http://miner4:7000")
    subnet.register_miner("miner-005", "resnet-50", "http://miner5:7000")
    
    # Print current matches
    for miner_id, validator_id in subnet.agent.matches.items():
        miner = subnet.agent.miners[miner_id]
        validator = subnet.agent.validators[validator_id]
        print(f"Miner {miner_id} ({miner.model_type.value}) matched with Validator {validator_id}")
    
    # Trigger validation for a miner
    result = await subnet.trigger_validation("miner-001")
    if result:
        print(f"Validation result: {result.score}")
    
    # Get emission allocations
    allocations = subnet.get_emission_allocation()
    print("Emission allocations:", allocations)
    
    # Get subnet status
    status = subnet.get_subnet_status()
    print("Subnet status:", status)


if __name__ == "__main__":
    asyncio.run(main())
