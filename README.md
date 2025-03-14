# Commune Subnet Agent

# Commune Subnet Agent

A Python implementation for a Commune subnet that intelligently matches miners offering different types of inference services to appropriate validators based on their namespace patterns. This subnet agent ensures that inference models are properly matched with validators specialized in scoring their particular type.

## 🌟 Features

- **Namespace-based Model Detection**: Automatically identifies model types from namespace strings
- **Intelligent Matching**: Pairs miners to specialized validators based on model type compatibility
- **Load Balancing**: Distributes miners across validators based on load and reliability
- **Dynamic Re-matching**: Automatically adapts when nodes join or leave the network
- **Emission Management**: Calculates token allocations based on quality scores and validation results
- **Modular and Extensible**: Built with a clean, object-oriented design that's easy to extend

## 📋 Requirements

- Python 3.8+
- asyncio

## 🚀 Getting Started

### Installation

```python
# Clone the repository
git clone https://github.com/huckl3berry/no-scope.git
cd no-scope

# Access the subnet agent code
python subnet_agent/subnet_agent.py
```

### Basic Usage

```python
import asyncio
from subnet_agent.subnet_agent import CommuneSubnet

async def main():
    # Create a subnet
    subnet = CommuneSubnet("my-inference-subnet")
    
    # Register validators with their specializations
    subnet.register_validator("val-001", "diffusion-verification", "diffusion", "http://validator1:8000")
    subnet.register_validator("val-002", "chat-verification", "chat", "http://validator2:8000")
    
    # Register miners with their namespaces
    subnet.register_miner("miner-001", "stable-diffusion-xl", "http://miner1:7000")
    subnet.register_miner("miner-002", "gpt-4-turbo", "http://miner2:7000")
    
    # Check matches
    for miner_id, validator_id in subnet.agent.matches.items():
        miner = subnet.agent.miners[miner_id]
        validator = subnet.agent.validators[validator_id]
        print(f"Miner {miner_id} ({miner.model_type.value}) matched with Validator {validator_id}")
    
    # Trigger validation
    result = await subnet.trigger_validation("miner-001")
    print(f"Validation score: {result.score}")
    
    # Get emission allocations
    allocations = subnet.get_emission_allocation()
    print("Emission allocations:", allocations)

if __name__ == "__main__":
    asyncio.run(main())
```

## 🏗️ Architecture

The subnet agent is built with a modular, class-based architecture:

### Core Components

- **ModelType**: Enum defining different types of inference models
- **NamespaceRegistry**: Maps namespace patterns to model types
- **SubnetAgent**: Core matching logic that links miners to validators
- **CommuneSubnet**: Main orchestration class managing the entire subnet

### Data Classes

- **MinerInfo**: Stores information about miners (ID, namespace, endpoint, etc.)
- **ValidatorInfo**: Stores information about validators (ID, specialization, capacity, etc.)
- **ValidationResult**: Records validation outcomes and quality scores

## 📊 Model Types Supported

The subnet currently supports matching for the following model types:

- **Diffusion Models**: Image generation (stable-diffusion, midjourney, dalle)
- **Chat Models**: Text generation and conversation (gpt, llama, claude, mistral)
- **Image Classification**: Computer vision (vit, resnet, yolo)
- **Audio Processing**: Speech recognition and audio analysis (whisper, wav2vec)
- **Embeddings**: Vector representations (sentence-transformer)
- **Video Generation**: Video creation models (sora)
- **Text-to-Speech**: Voice synthesis (eleven, tts)

## 🔄 Validation Flow

1. Miners register with the subnet, providing their namespace
2. Validators register with their specialization
3. The agent automatically matches miners to appropriate validators
4. Validation is triggered for miners
5. Validation results are stored and used for emission calculations
6. Token emissions are allocated based on quality scores

## 🧩 Integration with Commune

This subnet agent is designed to integrate with the wider Commune ecosystem:

1. **Substrate Chain**: Connect to the blockchain for on-chain validation recording
2. **App Store**: Make validated miners available through the Commune app store
3. **Value Capture**: Support token burning or USDC payments for accessing services

## 🛠️ Advanced Configuration

### Custom Namespace Mappings

You can register custom namespace mappings to the registry:

```python
subnet.agent.namespace_registry.register_namespace("my-custom-model", ModelType.DIFFUSION)
```

### Validator Capacity

Adjust how many miners each validator can handle:

```python
subnet.register_validator("val-001", "diffusion-verification", "diffusion", 
                         "http://validator1:8000", capacity=20)
```

## 📝 License

"Steal This Code" - This code is free to use, modify, and distribute however you want.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

Built with ❤️ for the Commune network
