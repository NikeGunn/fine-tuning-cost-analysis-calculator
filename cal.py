"""
Professional Fine-Tuning Cost Calculator
=========================================

A comprehensive tool for estimating fine-tuning costs across multiple cloud providers
and model configurations. Designed for daily use by ML engineers and developers.

Features:
- Support for AWS, HuggingFace, Google Cloud, Azure
- Comprehensive model database (7B to 405B+ models)
- Professional PDF report generation
- Scenario comparison and analysis
- Configuration save/load functionality
"""

import json
import os
import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import math

# Import PDF generator
try:
    from pdf_generator import PDFReportGenerator, install_reportlab, REPORTLAB_AVAILABLE
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    REPORTLAB_AVAILABLE = False


class CloudProvider(Enum):
    """Supported cloud providers for fine-tuning"""
    AWS_SAGEMAKER = "aws_sagemaker"
    HUGGINGFACE = "huggingface"
    GOOGLE_CLOUD = "google_cloud"
    AZURE_ML = "azure_ml"
    LAMBDA_LABS = "lambda_labs"
    RUNPOD = "runpod"


class ModelSize(Enum):
    """Standard model sizes"""
    SMALL_7B = "7B"
    MEDIUM_13B = "13B"
    LARGE_30B = "30B"
    XLARGE_70B = "70B"
    XXLARGE_405B = "405B"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    size: str
    parameters: int  # in billions
    memory_per_gpu: float  # GB
    recommended_gpus: int
    tokens_per_gpu_hour: float
    supports_qlora: bool = True
    supports_lora: bool = True
    supports_full_finetune: bool = True


@dataclass
class ProviderPricing:
    """Pricing configuration for a cloud provider"""
    name: str
    gpu_types: Dict[str, float]  # GPU type -> price per hour
    storage_cost_per_gb: float
    data_transfer_cost_per_gb: float
    additional_fees: float = 0.0  # monthly/setup fees


@dataclass
class CalculationInput:
    """Input parameters for cost calculation"""
    provider: CloudProvider
    model_config: ModelConfig
    gpu_type: str
    num_gpus: int
    training_method: str  # "qlora", "lora", "full"
    num_examples: int
    tokens_per_example: int
    epochs: int
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    sequence_length: int = 512
    storage_gb: float = 100.0
    data_transfer_gb: float = 10.0


@dataclass
class CalculationResult:
    """Result of cost calculation"""
    total_tokens: int
    effective_batch_size: int
    steps_per_epoch: int
    total_steps: int
    gpu_hours: float
    wallclock_hours: float
    compute_cost: float
    storage_cost: float
    data_transfer_cost: float
    additional_fees: float
    total_cost: float
    cost_per_token: float
    cost_per_example: float


class FineTuneCalculator:
    """Professional fine-tuning cost calculator"""

    def __init__(self):
        self.models = self._initialize_models()
        self.providers = self._initialize_providers()
        self.calculation_history: List[Dict] = []

    def _initialize_models(self) -> Dict[str, ModelConfig]:
        """Initialize comprehensive model database"""
        models = {
            # Llama Family
            "llama2-7b": ModelConfig("Llama 2 7B", "7B", 7, 16, 1, 5e7),
            "llama2-13b": ModelConfig("Llama 2 13B", "13B", 13, 32, 2, 3e7),
            "llama2-70b": ModelConfig("Llama 2 70B", "70B", 70, 80, 8, 7e6),
            "llama3-8b": ModelConfig("Llama 3 8B", "8B", 8, 18, 1, 4.5e7),
            "llama3-70b": ModelConfig("Llama 3 70B", "70B", 70, 80, 8, 7e6),
            "llama3.1-405b": ModelConfig("Llama 3.1 405B", "405B", 405, 200, 32, 2e6),

            # Mistral Family
            "mistral-7b": ModelConfig("Mistral 7B", "7B", 7, 16, 1, 5.2e7),
            "mixtral-8x7b": ModelConfig("Mixtral 8x7B", "56B", 56, 90, 4, 1.5e7),
            "mixtral-8x22b": ModelConfig("Mixtral 8x22B", "176B", 176, 150, 16, 3e6),

            # Code Models
            "codellama-7b": ModelConfig("CodeLlama 7B", "7B", 7, 16, 1, 4.8e7),
            "codellama-13b": ModelConfig("CodeLlama 13B", "13B", 13, 32, 2, 2.8e7),
            "codellama-34b": ModelConfig("CodeLlama 34B", "34B", 34, 70, 4, 1.2e7),

            # Falcon Family
            "falcon-7b": ModelConfig("Falcon 7B", "7B", 7, 16, 1, 4.9e7),
            "falcon-40b": ModelConfig("Falcon 40B", "40B", 40, 80, 4, 1e7),
            "falcon-180b": ModelConfig("Falcon 180B", "180B", 180, 160, 16, 2.5e6),

            # Gemma Family
            "gemma-2b": ModelConfig("Gemma 2B", "2B", 2, 8, 1, 8e7),
            "gemma-7b": ModelConfig("Gemma 7B", "7B", 7, 16, 1, 5.1e7),

            # Other Popular Models
            "qwen-7b": ModelConfig("Qwen 7B", "7B", 7, 16, 1, 4.9e7),
            "qwen-14b": ModelConfig("Qwen 14B", "14B", 14, 32, 2, 2.7e7),
            "qwen-72b": ModelConfig("Qwen 72B", "72B", 72, 85, 8, 6.8e6),
        }
        return models

    def _initialize_providers(self) -> Dict[CloudProvider, ProviderPricing]:
        """Initialize cloud provider pricing"""
        providers = {
            CloudProvider.AWS_SAGEMAKER: ProviderPricing(
                name="AWS SageMaker",
                gpu_types={
                    "ml.g4dn.xlarge": 0.526,     # T4 16GB
                    "ml.g4dn.2xlarge": 0.752,    # T4 16GB
                    "ml.g5.xlarge": 1.006,       # A10G 24GB
                    "ml.g5.2xlarge": 1.212,      # A10G 24GB
                    "ml.g5.4xlarge": 2.03,       # A10G 24GB
                    "ml.p3.2xlarge": 3.06,       # V100 16GB
                    "ml.p3.8xlarge": 12.24,      # 4x V100 16GB
                    "ml.p4d.24xlarge": 32.77,    # 8x A100 40GB
                    "ml.p5.48xlarge": 98.32,     # 8x H100 80GB
                },
                storage_cost_per_gb=0.12,
                data_transfer_cost_per_gb=0.09,
                additional_fees=0.0
            ),

            CloudProvider.HUGGINGFACE: ProviderPricing(
                name="HuggingFace Spaces",
                gpu_types={
                    "T4-small": 0.60,           # T4 16GB
                    "T4-medium": 0.90,          # T4 16GB
                    "A10G-small": 1.05,         # A10G 24GB
                    "A10G-large": 3.15,         # A10G 24GB
                    "A100-large": 4.13,         # A100 40GB
                },
                storage_cost_per_gb=0.15,
                data_transfer_cost_per_gb=0.0,  # Free egress
                additional_fees=0.0
            ),

            CloudProvider.GOOGLE_CLOUD: ProviderPricing(
                name="Google Cloud Vertex AI",
                gpu_types={
                    "nvidia-tesla-t4": 0.35,     # T4 16GB
                    "nvidia-tesla-v100": 2.48,   # V100 16GB
                    "nvidia-tesla-a100": 2.93,   # A100 40GB
                    "nvidia-a100-80gb": 3.67,    # A100 80GB
                    "nvidia-h100-80gb": 4.89,    # H100 80GB
                },
                storage_cost_per_gb=0.10,
                data_transfer_cost_per_gb=0.12,
                additional_fees=0.0
            ),

            CloudProvider.AZURE_ML: ProviderPricing(
                name="Azure Machine Learning",
                gpu_types={
                    "Standard_NC6s_v3": 3.06,    # V100 16GB
                    "Standard_NC12s_v3": 6.12,   # 2x V100 16GB
                    "Standard_NC24s_v3": 12.24,  # 4x V100 16GB
                    "Standard_ND40rs_v2": 22.03, # 8x V100 32GB
                    "Standard_NDm_A100_v4": 3.67, # A100 80GB
                },
                storage_cost_per_gb=0.045,
                data_transfer_cost_per_gb=0.087,
                additional_fees=0.0
            ),

            CloudProvider.LAMBDA_LABS: ProviderPricing(
                name="Lambda Labs",
                gpu_types={
                    "gpu_1x_a10": 0.60,         # A10 24GB
                    "gpu_1x_a6000": 0.80,       # RTX A6000 48GB
                    "gpu_1x_a100": 1.10,        # A100 40GB
                    "gpu_1x_a100_80gb": 1.40,   # A100 80GB
                    "gpu_1x_h100": 2.00,        # H100 80GB
                    "gpu_8x_a100": 8.80,        # 8x A100 40GB
                    "gpu_8x_h100": 16.00,       # 8x H100 80GB
                },
                storage_cost_per_gb=0.10,
                data_transfer_cost_per_gb=0.0,  # Free egress
                additional_fees=0.0
            ),

            CloudProvider.RUNPOD: ProviderPricing(
                name="RunPod",
                gpu_types={
                    "NVIDIA RTX A4000": 0.34,    # 16GB
                    "NVIDIA RTX A5000": 0.44,    # 24GB
                    "NVIDIA RTX A6000": 0.69,    # 48GB
                    "NVIDIA A40": 0.79,          # 48GB
                    "NVIDIA A100-40GB": 1.89,    # 40GB
                    "NVIDIA A100-80GB": 2.69,    # 80GB
                    "NVIDIA H100": 4.89,         # 80GB
                },
                storage_cost_per_gb=0.15,
                data_transfer_cost_per_gb=0.02,
                additional_fees=0.0
            )
        }
        return providers

    def calculate_cost(self, input_params: CalculationInput) -> CalculationResult:
        """Calculate fine-tuning costs based on input parameters"""

        # Get provider pricing
        provider_pricing = self.providers[input_params.provider]
        gpu_cost_per_hour = provider_pricing.gpu_types[input_params.gpu_type]

        # Calculate training metrics
        total_tokens = input_params.num_examples * input_params.tokens_per_example * input_params.epochs
        effective_batch_size = input_params.batch_size * input_params.gradient_accumulation_steps * input_params.num_gpus
        steps_per_epoch = math.ceil(input_params.num_examples / effective_batch_size)
        total_steps = steps_per_epoch * input_params.epochs

        # Adjust throughput based on training method
        base_throughput = input_params.model_config.tokens_per_gpu_hour
        if input_params.training_method == "qlora":
            throughput_multiplier = 1.0  # QLoRA is efficient
        elif input_params.training_method == "lora":
            throughput_multiplier = 0.85  # LoRA slightly slower
        else:  # full fine-tuning
            throughput_multiplier = 0.6  # Full fine-tuning much slower

        adjusted_throughput = base_throughput * throughput_multiplier

        # Calculate time and costs
        gpu_hours = total_tokens / adjusted_throughput
        wallclock_hours = gpu_hours / input_params.num_gpus

        compute_cost = gpu_hours * gpu_cost_per_hour
        storage_cost = input_params.storage_gb * provider_pricing.storage_cost_per_gb * (wallclock_hours / 24)  # Daily rate
        data_transfer_cost = input_params.data_transfer_gb * provider_pricing.data_transfer_cost_per_gb
        additional_fees = provider_pricing.additional_fees

        total_cost = compute_cost + storage_cost + data_transfer_cost + additional_fees
        cost_per_token = total_cost / total_tokens if total_tokens > 0 else 0
        cost_per_example = total_cost / input_params.num_examples if input_params.num_examples > 0 else 0

        return CalculationResult(
            total_tokens=total_tokens,
            effective_batch_size=effective_batch_size,
            steps_per_epoch=steps_per_epoch,
            total_steps=total_steps,
            gpu_hours=gpu_hours,
            wallclock_hours=wallclock_hours,
            compute_cost=compute_cost,
            storage_cost=storage_cost,
            data_transfer_cost=data_transfer_cost,
            additional_fees=additional_fees,
            total_cost=total_cost,
            cost_per_token=cost_per_token,
            cost_per_example=cost_per_example
        )

    def get_model_recommendations(self, num_examples: int, task_type: str = "general") -> List[str]:
        """Get model recommendations based on dataset size and task"""
        recommendations = []

        if num_examples < 1000:
            recommendations.extend(["llama2-7b", "mistral-7b", "gemma-7b"])
        elif num_examples < 10000:
            recommendations.extend(["llama2-13b", "llama3-8b", "codellama-13b"])
        elif num_examples < 100000:
            recommendations.extend(["llama2-70b", "mixtral-8x7b", "qwen-14b"])
        else:
            recommendations.extend(["llama3.1-405b", "mixtral-8x22b", "falcon-180b"])

        # Task-specific recommendations
        if task_type.lower() in ["code", "coding", "programming"]:
            recommendations = [m for m in recommendations if "code" in m.lower()] + \
                            [m for m in recommendations if "code" not in m.lower()]

        return recommendations[:5]  # Return top 5 recommendations

    def compare_scenarios(self, scenarios: List[CalculationInput]) -> Dict:
        """Compare multiple calculation scenarios"""
        results = []
        for scenario in scenarios:
            result = self.calculate_cost(scenario)
            results.append({
                "input": scenario,
                "result": result,
                "summary": {
                    "provider": scenario.provider.value,
                    "model": scenario.model_config.name,
                    "cost": result.total_cost,
                    "time_hours": result.wallclock_hours,
                    "cost_per_example": result.cost_per_example
                }
            })

        # Sort by cost
        results.sort(key=lambda x: x["result"].total_cost)

        return {
            "scenarios": results,
            "best_cost": results[0] if results else None,
            "best_time": min(results, key=lambda x: x["result"].wallclock_hours) if results else None,
            "comparison_summary": {
                "total_scenarios": len(results),
                "cost_range": (results[0]["result"].total_cost, results[-1]["result"].total_cost) if results else (0, 0),
                "time_range": (min(r["result"].wallclock_hours for r in results),
                             max(r["result"].wallclock_hours for r in results)) if results else (0, 0)
            }
        }

    def save_calculation(self, input_params: CalculationInput, result: CalculationResult, name: Optional[str] = None):
        """Save calculation to history"""
        calculation = {
            "name": name or f"Calculation_{len(self.calculation_history) + 1}",
            "timestamp": datetime.datetime.now().isoformat(),
            "input": asdict(input_params),
            "result": asdict(result)
        }
        self.calculation_history.append(calculation)
        return calculation

    def export_to_json(self, filename: Optional[str] = None):
        """Export calculation history to JSON"""
        if filename is None:
            filename = f"finetune_calculations_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        export_data = {
            "export_timestamp": datetime.datetime.now().isoformat(),
            "calculator_version": "1.0",
            "calculations": self.calculation_history
        }

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

        return filename

    def load_from_json(self, filename: str):
        """Load calculation history from JSON"""
        with open(filename, 'r') as f:
            data = json.load(f)

        self.calculation_history = data.get("calculations", [])
        return len(self.calculation_history)


class FineTuneCalculatorCLI:
    """Command-line interface for the fine-tuning calculator"""

    def __init__(self):
        self.calculator = FineTuneCalculator()
        self.current_scenario = None

    def run(self):
        """Main CLI loop"""
        print("🚀 Professional Fine-Tuning Cost Calculator")
        print("=" * 50)
        print("Welcome! This tool helps you estimate fine-tuning costs across multiple providers.")
        print()

        while True:
            self.show_main_menu()
            choice = input("\nSelect an option (1-8): ").strip()

            if choice == "1":
                self.calculate_single_scenario()
            elif choice == "2":
                self.compare_multiple_scenarios()
            elif choice == "3":
                self.show_model_recommendations()
            elif choice == "4":
                self.generate_pdf_report()
            elif choice == "5":
                self.save_load_configurations()
            elif choice == "6":
                self.show_calculation_history()
            elif choice == "7":
                self.show_provider_pricing()
            elif choice == "8":
                print("\n👋 Thanks for using the Fine-Tuning Cost Calculator!")
                break
            else:
                print("❌ Invalid option. Please try again.")

    def show_main_menu(self):
        """Display the main menu"""
        print("\n📋 Main Menu:")
        print("1. 💰 Calculate Single Scenario")
        print("2. 📊 Compare Multiple Scenarios")
        print("3. 🎯 Get Model Recommendations")
        print("4. 📄 Generate PDF Report")
        print("5. 💾 Save/Load Configurations")
        print("6. 📚 View Calculation History")
        print("7. 💵 View Provider Pricing")
        print("8. 🚪 Exit")

    def calculate_single_scenario(self):
        """Calculate cost for a single scenario"""
        print("\n💰 Single Scenario Calculation")
        print("-" * 30)

        try:
            # Get input parameters
            input_params = self.get_calculation_inputs()
            if input_params is None:
                return

            # Calculate
            result = self.calculator.calculate_cost(input_params)

            # Display results
            self.display_calculation_result(input_params, result)

            # Save option
            save = input("\n💾 Save this calculation? (y/n): ").strip().lower()
            if save == 'y':
                name = input("Enter a name for this calculation: ").strip()
                self.calculator.save_calculation(input_params, result, name)
                print("✅ Calculation saved!")

        except KeyboardInterrupt:
            print("\n❌ Calculation cancelled.")
        except Exception as e:
            print(f"❌ Error: {e}")

    def get_calculation_inputs(self) -> Optional[CalculationInput]:
        """Get calculation inputs from user"""
        try:
            # Provider selection
            print("\n🌐 Select Cloud Provider:")
            providers = list(CloudProvider)
            for i, provider in enumerate(providers, 1):
                print(f"{i}. {self.calculator.providers[provider].name}")

            provider_idx = int(input("Provider choice (1-{}): ".format(len(providers)))) - 1
            if provider_idx < 0 or provider_idx >= len(providers):
                raise ValueError("Invalid provider selection")
            provider = providers[provider_idx]

            # Model selection
            print("\n🤖 Select Model:")
            models = list(self.calculator.models.items())
            for i, (key, model) in enumerate(models, 1):
                print(f"{i}. {model.name} ({model.size}, {model.parameters}B params)")

            model_idx = int(input("Model choice (1-{}): ".format(len(models)))) - 1
            if model_idx < 0 or model_idx >= len(models):
                raise ValueError("Invalid model selection")
            model_config = models[model_idx][1]

            # GPU type selection
            print("\n🎮 Select GPU Type:")
            gpu_types = list(self.calculator.providers[provider].gpu_types.items())
            for i, (gpu_type, price) in enumerate(gpu_types, 1):
                print(f"{i}. {gpu_type} (${price:.2f}/hour)")

            gpu_idx = int(input("GPU choice (1-{}): ".format(len(gpu_types)))) - 1
            if gpu_idx < 0 or gpu_idx >= len(gpu_types):
                raise ValueError("Invalid GPU selection")
            gpu_type = gpu_types[gpu_idx][0]

            # Training method
            print("\n⚙️ Select Training Method:")
            print("1. QLoRA (Quantized LoRA - Most efficient)")
            print("2. LoRA (Low-Rank Adaptation)")
            print("3. Full Fine-tuning (Highest quality, most expensive)")

            method_choice = int(input("Training method (1-3): "))
            methods = ["qlora", "lora", "full"]
            if method_choice < 1 or method_choice > 3:
                raise ValueError("Invalid training method")
            training_method = methods[method_choice - 1]

            # Basic parameters
            print("\n📊 Training Parameters:")
            num_examples = int(input("Number of training examples: "))
            tokens_per_example = int(input("Average tokens per example (default 512): ") or "512")
            epochs = int(input("Number of epochs (default 3): ") or "3")

            # Advanced parameters
            print("\n🔧 Advanced Settings (press Enter for defaults):")
            batch_size = int(input("Batch size per GPU (default 4): ") or "4")
            gradient_accumulation_steps = int(input("Gradient accumulation steps (default 4): ") or "4")
            sequence_length = int(input("Sequence length (default 512): ") or "512")

            # GPU count (auto-recommend but allow override)
            recommended_gpus = model_config.recommended_gpus
            num_gpus = int(input(f"Number of GPUs (recommended {recommended_gpus}): ") or str(recommended_gpus))

            # Storage and transfer
            storage_gb = float(input("Storage needed in GB (default 100): ") or "100")
            data_transfer_gb = float(input("Data transfer in GB (default 10): ") or "10")

            return CalculationInput(
                provider=provider,
                model_config=model_config,
                gpu_type=gpu_type,
                num_gpus=num_gpus,
                training_method=training_method,
                num_examples=num_examples,
                tokens_per_example=tokens_per_example,
                epochs=epochs,
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                sequence_length=sequence_length,
                storage_gb=storage_gb,
                data_transfer_gb=data_transfer_gb
            )

        except (ValueError, KeyboardInterrupt) as e:
            print(f"❌ Invalid input: {e}")
            return None

    def display_calculation_result(self, input_params: CalculationInput, result: CalculationResult):
        """Display calculation results in a formatted way"""
        print("\n" + "="*60)
        print("📊 CALCULATION RESULTS")
        print("="*60)

        # Configuration summary
        print(f"\n🔧 Configuration:")
        print(f"   Provider: {self.calculator.providers[input_params.provider].name}")
        print(f"   Model: {input_params.model_config.name}")
        print(f"   GPU: {input_params.gpu_type} × {input_params.num_gpus}")
        print(f"   Training: {input_params.training_method.upper()}")
        print(f"   Dataset: {input_params.num_examples:,} examples × {input_params.epochs} epochs")

        # Training metrics
        print(f"\n📈 Training Metrics:")
        print(f"   Total tokens: {result.total_tokens:,}")
        print(f"   Effective batch size: {result.effective_batch_size}")
        print(f"   Steps per epoch: {result.steps_per_epoch:,}")
        print(f"   Total training steps: {result.total_steps:,}")

        # Time estimates
        print(f"\n⏱️ Time Estimates:")
        print(f"   GPU hours (total): {result.gpu_hours:.1f}")
        print(f"   Wallclock time: {result.wallclock_hours:.1f} hours")
        if result.wallclock_hours > 24:
            days = result.wallclock_hours / 24
            print(f"   Wallclock time: {days:.1f} days")

        # Cost breakdown
        print(f"\n💰 Cost Breakdown:")
        print(f"   Compute cost: ${result.compute_cost:.2f}")
        print(f"   Storage cost: ${result.storage_cost:.2f}")
        print(f"   Data transfer: ${result.data_transfer_cost:.2f}")
        if result.additional_fees > 0:
            print(f"   Additional fees: ${result.additional_fees:.2f}")
        print(f"   " + "-"*25)
        print(f"   TOTAL COST: ${result.total_cost:.2f}")

        # Per-unit costs
        print(f"\n📊 Per-Unit Costs:")
        print(f"   Cost per token: ${result.cost_per_token:.6f}")
        print(f"   Cost per example: ${result.cost_per_example:.4f}")

        print("="*60)

    def compare_multiple_scenarios(self):
        """Compare multiple scenarios"""
        print("\n📊 Compare Multiple Scenarios")
        print("-" * 30)

        scenarios = []
        while True:
            print(f"\n🔢 Scenario {len(scenarios) + 1}:")
            input_params = self.get_calculation_inputs()
            if input_params is None:
                break
            scenarios.append(input_params)

            if len(scenarios) >= 5:
                print("⚠️ Maximum 5 scenarios reached.")
                break

            add_more = input("\nAdd another scenario? (y/n): ").strip().lower()
            if add_more != 'y':
                break

        if len(scenarios) < 2:
            print("❌ Need at least 2 scenarios to compare.")
            return

        # Compare scenarios
        comparison = self.calculator.compare_scenarios(scenarios)

        # Display comparison
        print("\n" + "="*80)
        print("📊 SCENARIO COMPARISON")
        print("="*80)

        for i, scenario_data in enumerate(comparison["scenarios"], 1):
            result = scenario_data["result"]
            summary = scenario_data["summary"]
            print(f"\n🔢 Scenario {i}: {summary['model']} on {summary['provider']}")
            print(f"   Cost: ${result.total_cost:.2f}")
            print(f"   Time: {result.wallclock_hours:.1f} hours")
            print(f"   Cost/example: ${result.cost_per_example:.4f}")

        # Best options
        best_cost = comparison["best_cost"]
        best_time = comparison["best_time"]

        print(f"\n🏆 Best Cost: {best_cost['summary']['model']} - ${best_cost['result'].total_cost:.2f}")
        print(f"🏆 Best Time: {best_time['summary']['model']} - {best_time['result'].wallclock_hours:.1f} hours")

        # Save comparison
        save = input("\n💾 Save this comparison? (y/n): ").strip().lower()
        if save == 'y':
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"scenario_comparison_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(comparison, f, indent=2, default=str)
            print(f"✅ Comparison saved to {filename}")

    def show_model_recommendations(self):
        """Show model recommendations based on dataset size"""
        print("\n🎯 Model Recommendations")
        print("-" * 25)

        try:
            num_examples = int(input("Number of training examples: "))
            task_type = input("Task type (general/code/chat): ").strip() or "general"

            recommendations = self.calculator.get_model_recommendations(num_examples, task_type)

            print(f"\n📋 Recommended models for {num_examples:,} examples ({task_type}):")
            for i, model_key in enumerate(recommendations, 1):
                model = self.calculator.models[model_key]
                print(f"{i}. {model.name}")
                print(f"   Size: {model.size} ({model.parameters}B parameters)")
                print(f"   Memory: {model.memory_per_gpu}GB per GPU")
                print(f"   Recommended GPUs: {model.recommended_gpus}")
                print()

        except ValueError:
            print("❌ Invalid input.")

    def generate_pdf_report(self):
        """Generate PDF report using professional PDF generator"""
        print("\n📄 Generate PDF Report")
        print("-" * 20)

        if not self.calculator.calculation_history:
            print("❌ No calculations found. Run some calculations first.")
            return

        # Check if PDF generation is available
        if not PDF_AVAILABLE or not REPORTLAB_AVAILABLE:
            print("⚠️ PDF generation requires ReportLab library.")
            install_choice = input("Install ReportLab now? (y/n): ").strip().lower()
            if install_choice == 'y':
                if install_reportlab():
                    print("✅ ReportLab installed! Please restart the application to use PDF features.")
                else:
                    print("❌ Installation failed. Generating text reports instead.")
                    self._generate_text_reports()
                return
            else:
                print("Generating text reports instead...")
                self._generate_text_reports()
                return

        try:
            pdf_generator = PDFReportGenerator()

            print("📊 Available PDF reports:")
            print("1. Single calculation detailed report")
            print("2. Multiple calculations summary")
            print("3. Executive summary for CEO")
            print("4. Scenario comparison report")

            choice = input("Select report type (1-4): ").strip()

            if choice == "1":
                self._generate_single_pdf_report(pdf_generator)
            elif choice == "2":
                self._generate_summary_pdf_report(pdf_generator)
            elif choice == "3":
                self._generate_executive_pdf_report(pdf_generator)
            elif choice == "4":
                self._generate_comparison_pdf_report(pdf_generator)
            else:
                print("❌ Invalid choice.")

        except Exception as e:
            print(f"❌ PDF generation error: {e}")
            print("Falling back to text reports...")
            self._generate_text_reports()

    def _generate_single_pdf_report(self, pdf_generator):
        """Generate single calculation PDF report"""
        print("\n� Select calculation:")
        for i, calc in enumerate(self.calculator.calculation_history, 1):
            print(f"{i}. {calc['name']} - {calc['timestamp'][:10]}")

        try:
            choice = int(input("Select calculation: ")) - 1
            if 0 <= choice < len(self.calculator.calculation_history):
                calc = self.calculator.calculation_history[choice]
                filename = f"detailed_report_{calc['name'].replace(' ', '_')}.pdf"

                generated_file = pdf_generator.generate_detailed_report(calc, filename)
                print(f"✅ Detailed PDF report generated: {generated_file}")
            else:
                print("❌ Invalid selection.")
        except (ValueError, IndexError):
            print("❌ Invalid input.")

    def _generate_summary_pdf_report(self, pdf_generator):
        """Generate summary PDF report for multiple calculations"""
        filename = f"calculations_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        # For summary, we'll use the executive summary generator with all calculations
        generated_file = pdf_generator.generate_executive_summary(self.calculator.calculation_history, filename)
        print(f"✅ Summary PDF report generated: {generated_file}")

    def _generate_executive_pdf_report(self, pdf_generator):
        """Generate executive summary PDF for CEO"""
        filename = f"executive_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        generated_file = pdf_generator.generate_executive_summary(self.calculator.calculation_history, filename)
        print(f"✅ Executive PDF summary generated: {generated_file}")
        print("📧 This professional report is ready to share with stakeholders!")

    def _generate_comparison_pdf_report(self, pdf_generator):
        """Generate scenario comparison PDF report"""
        if len(self.calculator.calculation_history) < 2:
            print("❌ Need at least 2 calculations for comparison report.")
            return

        # Create comparison data structure
        scenarios = []
        for calc in self.calculator.calculation_history:
            scenarios.append({
                "input": calc["input"],
                "result": calc["result"],
                "summary": {
                    "provider": calc["input"].get("provider", "Unknown"),
                    "model": calc["input"].get("model_config", {}).get("name", "Unknown"),
                    "cost": calc["result"]["total_cost"],
                    "time_hours": calc["result"]["wallclock_hours"],
                    "cost_per_example": calc["result"]["cost_per_example"]
                }
            })

        # Sort by cost
        scenarios.sort(key=lambda x: x["result"]["total_cost"])

        comparison_data = {
            "scenarios": scenarios,
            "best_cost": scenarios[0] if scenarios else None,
            "best_time": min(scenarios, key=lambda x: x["result"]["wallclock_hours"]) if scenarios else None,
            "comparison_summary": {
                "total_scenarios": len(scenarios),
                "cost_range": (scenarios[0]["result"]["total_cost"], scenarios[-1]["result"]["total_cost"]) if scenarios else (0, 0),
                "time_range": (min(s["result"]["wallclock_hours"] for s in scenarios),
                             max(s["result"]["wallclock_hours"] for s in scenarios)) if scenarios else (0, 0)
            }
        }

        filename = f"scenario_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        generated_file = pdf_generator.generate_comparison_report(comparison_data, filename)
        print(f"✅ Comparison PDF report generated: {generated_file}")

    def _generate_text_reports(self):
        """Fallback text report generation when PDF is not available"""
        print("📊 Available text reports:")
        print("1. Single calculation report")
        print("2. Multiple calculations summary")
        print("3. Executive summary for CEO")

        choice = input("Select report type (1-3): ").strip()

        if choice == "1":
            self._generate_single_report()
        elif choice == "2":
            self._generate_summary_report()
        elif choice == "3":
            self._generate_executive_report()
        else:
            print("❌ Invalid choice.")

    def _generate_single_report(self):
        """Generate single calculation PDF report"""
        print("\n📋 Select calculation:")
        for i, calc in enumerate(self.calculator.calculation_history, 1):
            print(f"{i}. {calc['name']} - {calc['timestamp'][:10]}")

        try:
            choice = int(input("Select calculation: ")) - 1
            if 0 <= choice < len(self.calculator.calculation_history):
                calc = self.calculator.calculation_history[choice]
                filename = f"finetune_report_{calc['name'].replace(' ', '_')}.txt"

                # For now, generate text report (PDF implementation coming next)
                with open(filename, 'w') as f:
                    f.write("FINE-TUNING COST ANALYSIS REPORT\n")
                    f.write("=" * 40 + "\n\n")
                    f.write(f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Calculation: {calc['name']}\n")
                    f.write(f"Original Date: {calc['timestamp'][:10]}\n\n")

                    # Add detailed analysis
                    result = calc['result']
                    f.write("COST SUMMARY\n")
                    f.write("-" * 15 + "\n")
                    f.write(f"Total Cost: ${result['total_cost']:.2f}\n")
                    f.write(f"Compute Cost: ${result['compute_cost']:.2f}\n")
                    f.write(f"Storage Cost: ${result['storage_cost']:.2f}\n")
                    f.write(f"Data Transfer: ${result['data_transfer_cost']:.2f}\n\n")

                    f.write("TIME ANALYSIS\n")
                    f.write("-" * 15 + "\n")
                    f.write(f"Total GPU Hours: {result['gpu_hours']:.1f}\n")
                    f.write(f"Wallclock Time: {result['wallclock_hours']:.1f} hours\n")
                    f.write(f"Training Steps: {result['total_steps']:,}\n\n")

                    f.write("EFFICIENCY METRICS\n")
                    f.write("-" * 18 + "\n")
                    f.write(f"Cost per Token: ${result['cost_per_token']:.6f}\n")
                    f.write(f"Cost per Example: ${result['cost_per_example']:.4f}\n")

                print(f"✅ Report saved to {filename}")
            else:
                print("❌ Invalid selection.")
        except (ValueError, IndexError):
            print("❌ Invalid input.")

    def _generate_summary_report(self):
        """Generate summary report for multiple calculations"""
        filename = f"finetune_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(filename, 'w') as f:
            f.write("FINE-TUNING CALCULATIONS SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Calculations: {len(self.calculator.calculation_history)}\n\n")

            total_cost = sum(calc['result']['total_cost'] for calc in self.calculator.calculation_history)
            avg_cost = total_cost / len(self.calculator.calculation_history) if self.calculator.calculation_history else 0

            f.write("COST ANALYSIS\n")
            f.write("-" * 15 + "\n")
            f.write(f"Total Cost (all calculations): ${total_cost:.2f}\n")
            f.write(f"Average Cost per Calculation: ${avg_cost:.2f}\n")
            f.write(f"Min Cost: ${min(calc['result']['total_cost'] for calc in self.calculator.calculation_history):.2f}\n")
            f.write(f"Max Cost: ${max(calc['result']['total_cost'] for calc in self.calculator.calculation_history):.2f}\n\n")

            f.write("INDIVIDUAL CALCULATIONS\n")
            f.write("-" * 25 + "\n")
            for calc in self.calculator.calculation_history:
                f.write(f"• {calc['name']}: ${calc['result']['total_cost']:.2f} ({calc['timestamp'][:10]})\n")

        print(f"✅ Summary report saved to {filename}")

    def _generate_executive_report(self):
        """Generate executive summary for CEO"""
        filename = f"executive_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(filename, 'w') as f:
            f.write("EXECUTIVE SUMMARY - FINE-TUNING COST ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Prepared by: ML Engineering Team\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%B %d, %Y')}\n\n")

            if not self.calculator.calculation_history:
                f.write("No calculations performed yet.\n")
                return

            total_cost = sum(calc['result']['total_cost'] for calc in self.calculator.calculation_history)
            avg_cost = total_cost / len(self.calculator.calculation_history)

            f.write("KEY FINDINGS\n")
            f.write("-" * 12 + "\n")
            f.write(f"• Analyzed {len(self.calculator.calculation_history)} fine-tuning scenarios\n")
            f.write(f"• Total estimated cost: ${total_cost:.2f}\n")
            f.write(f"• Average cost per scenario: ${avg_cost:.2f}\n")
            f.write(f"• Cost range: ${min(calc['result']['total_cost'] for calc in self.calculator.calculation_history):.2f} - ${max(calc['result']['total_cost'] for calc in self.calculator.calculation_history):.2f}\n\n")

            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            f.write("• Consider QLoRA for cost optimization (up to 60% savings)\n")
            f.write("• Smaller models (7B-13B) suitable for most tasks\n")
            f.write("• Lambda Labs and RunPod offer competitive pricing\n")
            f.write("• Plan for 2-7 days training time for large models\n\n")

            f.write("NEXT STEPS\n")
            f.write("-" * 10 + "\n")
            f.write("1. Approve budget allocation based on cost estimates\n")
            f.write("2. Select optimal provider and model configuration\n")
            f.write("3. Begin pilot fine-tuning project\n")
            f.write("4. Monitor actual vs. estimated costs\n")

        print(f"✅ Executive summary saved to {filename}")

    def save_load_configurations(self):
        """Save or load calculator configurations"""
        print("\n💾 Configuration Management")
        print("-" * 25)
        print("1. Save current calculations")
        print("2. Load previous calculations")
        print("3. Export to JSON")

        choice = input("Select option (1-3): ").strip()

        if choice == "1":
            filename = input("Enter filename (or press Enter for auto): ").strip()
            if not filename:
                filename = self.calculator.export_to_json()
            else:
                filename = self.calculator.export_to_json(filename)
            print(f"✅ Calculations saved to {filename}")

        elif choice == "2":
            filename = input("Enter filename to load: ").strip()
            if os.path.exists(filename):
                count = self.calculator.load_from_json(filename)
                print(f"✅ Loaded {count} calculations from {filename}")
            else:
                print("❌ File not found.")

        elif choice == "3":
            filename = self.calculator.export_to_json()
            print(f"✅ Data exported to {filename}")
        else:
            print("❌ Invalid choice.")

    def show_calculation_history(self):
        """Show calculation history"""
        print("\n📚 Calculation History")
        print("-" * 20)

        if not self.calculator.calculation_history:
            print("No calculations found.")
            return

        for i, calc in enumerate(self.calculator.calculation_history, 1):
            print(f"\n{i}. {calc['name']}")
            print(f"   Date: {calc['timestamp'][:10]}")
            print(f"   Cost: ${calc['result']['total_cost']:.2f}")
            print(f"   Time: {calc['result']['wallclock_hours']:.1f} hours")

    def show_provider_pricing(self):
        """Show current provider pricing"""
        print("\n💵 Provider Pricing")
        print("-" * 18)

        for provider, pricing in self.calculator.providers.items():
            print(f"\n🌐 {pricing.name}:")
            print("   GPU Options:")
            for gpu_type, price in pricing.gpu_types.items():
                print(f"     • {gpu_type}: ${price:.2f}/hour")
            print(f"   Storage: ${pricing.storage_cost_per_gb:.3f}/GB/day")
            print(f"   Data Transfer: ${pricing.data_transfer_cost_per_gb:.3f}/GB")


# Main execution
if __name__ == "__main__":
    try:
        cli = FineTuneCalculatorCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please report this issue to the development team.")
