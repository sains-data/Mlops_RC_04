"""
Command Line Interface for MLOps Pothole Detection
"""

import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.ingestion import DataIngestion
from src.data.analysis import DataAnalyzer
from src.training.train import ModelTrainer
from src.training.tuning import HyperparameterTuner
from src.evaluation.evaluate import ModelEvaluator

app = typer.Typer(help="MLOps CLI for Pothole Detection")
console = Console()


@app.command()
def validate_data(dataset_path: str = "dataset"):
    """
    Validate dataset structure and integrity
    
    Args:
        dataset_path: Path to dataset directory
    """
    console.print("\n[bold blue]🔍 Starting Data Validation[/bold blue]\n")
    
    ingestion = DataIngestion(dataset_path)
    is_valid = ingestion.run_full_validation()
    
    if is_valid:
        console.print("\n[bold green]✓ Dataset validation successful![/bold green]\n")
    else:
        console.print("\n[bold red]✗ Dataset validation failed! Please fix issues.[/bold red]\n")


@app.command()
def analyze_data(dataset_path: str = "dataset"):
    """
    Perform Exploratory Data Analysis on dataset
    
    Args:
        dataset_path: Path to dataset directory
    """
    console.print("\n[bold blue]📊 Starting Data Analysis[/bold blue]\n")
    
    analyzer = DataAnalyzer(dataset_path)
    report = analyzer.generate_eda_report()
    
    console.print("\n[bold green]✓ EDA completed! Check outputs/eda/ for results[/bold green]\n")


@app.command()
def train(
    model: str = typer.Option("yolov8n", help="Model type (yolov8n, yolov8s)"),
    epochs: int = typer.Option(None, help="Number of epochs"),
    batch: int = typer.Option(None, help="Batch size"),
    data_yaml: str = typer.Option("configs/data.yaml", help="Path to data.yaml")
):
    """
    Train a YOLO model
    
    Args:
        model: Model type
        epochs: Number of epochs
        batch: Batch size
        data_yaml: Path to data.yaml
    """
    console.print(f"\n[bold blue]🚀 Starting Training: {model}[/bold blue]\n")
    
    trainer = ModelTrainer()
    result = trainer.train_model(
        model_type=model,
        epochs=epochs,
        batch_size=batch,
        data_yaml=data_yaml
    )
    
    console.print("\n[bold green]✓ Training completed![/bold green]")
    console.print(f"Model saved to: {result['model_path']}")
    console.print(f"mAP@0.5: {result['map50']:.4f}")
    console.print(f"mAP@0.5:0.95: {result['map50_95']:.4f}\n")


@app.command()
def train_all():
    """Train all configured models"""
    console.print("\n[bold blue]🚀 Training All Models[/bold blue]\n")
    
    trainer = ModelTrainer()
    results = trainer.train_multiple_models()
    
    # Display results table
    table = Table(title="Training Results")
    table.add_column("Model", style="cyan")
    table.add_column("mAP@0.5", style="green")
    table.add_column("mAP@0.5:0.95", style="green")
    table.add_column("Status", style="yellow")
    
    for model_name, result in results.items():
        if "error" in result:
            table.add_row(model_name, "-", "-", "Failed")
        else:
            table.add_row(
                model_name,
                f"{result['map50']:.4f}",
                f"{result['map50_95']:.4f}",
                "Success"
            )
    
    console.print(table)
    console.print("\n[bold green]✓ All models trained![/bold green]\n")


@app.command()
def evaluate(
    model_path: str = typer.Argument(..., help="Path to model file"),
    split: str = typer.Option("val", help="Dataset split (val/test)")
):
    """
    Evaluate a trained model
    
    Args:
        model_path: Path to model file
        split: Dataset split to evaluate on
    """
    console.print(f"\n[bold blue]📈 Evaluating Model: {model_path}[/bold blue]\n")
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(model_path, split=split)
    
    # Display metrics
    table = Table(title="Evaluation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for metric, value in metrics.items():
        table.add_row(metric.upper(), f"{value:.4f}")
    
    console.print(table)
    console.print()


@app.command()
def test(model_path: str = typer.Argument(..., help="Path to model file")):
    """
    Test model on test set
    
    Args:
        model_path: Path to model file
    """
    console.print(f"\n[bold blue]🧪 Testing Model: {model_path}[/bold blue]\n")
    
    evaluator = ModelEvaluator()
    metrics = evaluator.test_model(model_path)
    
    # Display metrics
    table = Table(title="Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for metric, value in metrics.items():
        table.add_row(metric.upper(), f"{value:.4f}")
    
    console.print(table)
    console.print()


@app.command()
def tune(
    model: str = typer.Option("yolov8n", help="Model type"),
    n_trials: int = typer.Option(50, help="Number of trials"),
    timeout: int = typer.Option(3600, help="Timeout in seconds")
):
    """
    Run hyperparameter tuning
    
    Args:
        model: Model type
        n_trials: Number of trials
        timeout: Timeout in seconds
    """
    console.print(f"\n[bold blue]🎯 Hyperparameter Tuning: {model}[/bold blue]\n")
    
    tuner = HyperparameterTuner()
    results = tuner.tune(model_type=model, n_trials=n_trials, timeout=timeout)
    
    console.print("\n[bold green]✓ Tuning completed![/bold green]")
    console.print(f"Best mAP@0.5: {results['best_value']:.4f}")
    console.print("\nBest parameters:")
    
    for param, value in results['best_params'].items():
        console.print(f"  {param}: {value}")
    
    console.print()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    reload: bool = typer.Option(False, help="Enable auto-reload")
):
    """
    Start FastAPI server
    
    Args:
        host: Host to bind
        port: Port to bind
        reload: Enable auto-reload
    """
    console.print(f"\n[bold blue]🚀 Starting API Server[/bold blue]\n")
    console.print(f"Server: http://{host}:{port}")
    console.print(f"Docs: http://{host}:{port}/docs\n")
    
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload
    )


@app.command()
def compare_models():
    """Compare all trained models"""
    console.print("\n[bold blue]📊 Comparing Models[/bold blue]\n")
    
    evaluator = ModelEvaluator()
    
    # Find all trained models from runs/train/*/weights/best.pt
    available_models = {}
    runs_dir = Path("runs/train")
    
    if runs_dir.exists():
        for experiment_dir in runs_dir.iterdir():
            if experiment_dir.is_dir():
                best_model = experiment_dir / "weights" / "best.pt"
                if best_model.exists():
                    # Use experiment folder name as model identifier
                    model_name = experiment_dir.name
                    available_models[model_name] = str(best_model)
                    console.print(f"[green]✓[/green] Found model: {model_name}")
    
    # Also check models/ directory
    models_dir = Path("models")
    if models_dir.exists():
        for model_file in models_dir.glob("*_best.pt"):
            model_name = model_file.stem.replace("_best", "")
            if model_name not in available_models:
                available_models[model_name] = str(model_file)
                console.print(f"[green]✓[/green] Found model: {model_name} (from models/)")
    
    if not available_models:
        console.print("[bold red]No trained models found![/bold red]\n")
        console.print("Train a model first using: [cyan]python cli.py train --model yolov8n[/cyan]\n")
        return
    
    comparison = evaluator.compare_models(available_models)
    
    # Display comparison table
    table = Table(title="Model Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("mAP@0.5", style="green")
    table.add_column("mAP@0.5:0.95", style="green")
    table.add_column("Precision", style="yellow")
    table.add_column("Recall", style="yellow")
    table.add_column("F1-Score", style="magenta")
    
    for model_name, metrics in comparison.items():
        table.add_row(
            model_name,
            f"{metrics['map50']:.4f}",
            f"{metrics['map50_95']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1_score']:.4f}"
        )
    
    console.print(table)
    console.print("\n[bold green]✓ Comparison plot saved to outputs/evaluation/[/bold green]\n")


if __name__ == "__main__":
    app()
