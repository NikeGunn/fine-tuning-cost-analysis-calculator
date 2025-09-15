# Professional Fine-Tuning Cost Calculator

## 🚀 Quick Start Guide

This is your comprehensive fine-tuning cost calculator designed for daily use by ML engineers and developers. It supports multiple cloud providers, model configurations, and generates professional PDF reports for stakeholders.

### Features
- **Multi-Provider Support**: AWS SageMaker, HuggingFace, Google Cloud, Azure ML, Lambda Labs, RunPod
- **Comprehensive Model Database**: 20+ popular models (7B to 405B parameters)
- **Professional PDF Reports**: Executive summaries, detailed analysis, scenario comparisons
- **Flexible Training Methods**: QLoRA, LoRA, Full Fine-tuning
- **Cost Optimization**: Smart recommendations and efficiency metrics
- **Configuration Management**: Save/load scenarios for daily use

### Installation

1. **Install dependencies** (optional for PDF generation):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the calculator**:
   ```bash
   python cal.py
   ```

### Daily Usage Workflow

1. **Start the Calculator**: Run `python cal.py`
2. **Calculate Scenario**: Choose option 1 to calculate costs for your specific needs
3. **Compare Options**: Use option 2 to compare multiple providers/models
4. **Generate Reports**: Create professional PDFs for your CEO/stakeholders
5. **Save Configurations**: Store frequently used setups for quick access

### Supported Models

#### Llama Family
- Llama 2 7B/13B/70B
- Llama 3 8B/70B
- Llama 3.1 405B

#### Mistral Family
- Mistral 7B
- Mixtral 8x7B/8x22B

#### Code Models
- CodeLlama 7B/13B/34B

#### Other Popular Models
- Falcon 7B/40B/180B
- Gemma 2B/7B
- Qwen 7B/14B/72B

### Cloud Providers & Pricing

| Provider | GPU Options | Storage | Data Transfer |
|----------|-------------|---------|---------------|
| AWS SageMaker | T4, A10G, V100, A100, H100 | $0.12/GB | $0.09/GB |
| HuggingFace | T4, A10G, A100 | $0.15/GB | Free |
| Google Cloud | T4, V100, A100, H100 | $0.10/GB | $0.12/GB |
| Azure ML | V100, A100 | $0.045/GB | $0.087/GB |
| Lambda Labs | A10, A6000, A100, H100 | $0.10/GB | Free |
| RunPod | RTX A4000-H100 | $0.15/GB | $0.02/GB |

### Training Methods

1. **QLoRA** (Recommended)
   - Most cost-effective (up to 60% savings)
   - Maintains 99% of full fine-tuning quality
   - Best for most applications

2. **LoRA**
   - Good balance of cost and quality
   - 15% slower than QLoRA
   - Suitable for performance-critical applications

3. **Full Fine-tuning**
   - Highest quality but most expensive
   - 40% slower and much more memory-intensive
   - Only for specialized use cases

### Cost Optimization Tips

1. **Start Small**: Begin with 7B models for proof of concept
2. **Use QLoRA**: 60% cost reduction with minimal quality loss
3. **Choose Lambda Labs/RunPod**: Often 30-50% cheaper than major clouds
4. **Optimize Batch Size**: Higher batch sizes = better GPU utilization
5. **Plan Training Time**: Consider overnight/weekend training for large models

### PDF Reports

The calculator generates three types of professional reports:

1. **Executive Summary**: High-level overview for CEOs and stakeholders
2. **Detailed Analysis**: Technical report with complete metrics
3. **Scenario Comparison**: Side-by-side analysis of multiple options

All reports include:
- Cost breakdowns with charts
- Time estimates and efficiency metrics
- Strategic recommendations
- Professional formatting for presentations

### Example Use Cases

#### Startup MVP
```
Model: Llama 2 7B
Provider: RunPod
GPU: RTX A6000
Training: QLoRA
Dataset: 1,000 examples
Estimated Cost: ~$15-25
Time: 2-4 hours
```

#### Enterprise Production
```
Model: Llama 3 70B
Provider: AWS SageMaker
GPU: 8x A100
Training: QLoRA
Dataset: 100,000 examples
Estimated Cost: ~$800-1,200
Time: 24-48 hours
```

#### Research/Experimentation
```
Model: Mixtral 8x7B
Provider: Lambda Labs
GPU: 4x A100
Training: LoRA
Dataset: 10,000 examples
Estimated Cost: ~$150-250
Time: 8-12 hours
```

### Support

For technical support or feature requests, contact your ML Engineering team.

---

**Version**: 1.0
**Last Updated**: September 2025
**Designed for**: Daily use by ML developers and fine-tuning specialists
