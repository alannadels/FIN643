"""
Main script to run the full RL options trading pipeline
"""

import argparse
from pathlib import Path


def run_pipeline(train_only=False, eval_only=False, viz_only=False):
    """
    Run the complete pipeline

    Args:
        train_only: Only train the model
        eval_only: Only evaluate (requires trained model)
        viz_only: Only create visualizations (requires evaluation results)
    """
    print("=" * 80)
    print("RL OPTIONS TRADING STRATEGY - FULL PIPELINE")
    print("=" * 80)

    if viz_only:
        print("\nCreating visualizations...")
        from visualize import create_visualizations
        create_visualizations()
        print("\nVisualization complete!")
        return

    if eval_only:
        print("\nRunning evaluation only...")
        from evaluate import main as evaluate_main
        evaluate_main()

        print("\nCreating visualizations...")
        from visualize import create_visualizations
        create_visualizations()

        print("\n" + "=" * 80)
        print("Evaluation complete!")
        print("=" * 80)
        return

    # Full pipeline or train only
    if not train_only:
        print("\nStep 1/3: Training RL agent")
        print("-" * 80)

    from train import train_agent
    model = train_agent()

    if train_only:
        print("\n" + "=" * 80)
        print("Training complete!")
        print("=" * 80)
        return

    print("\nStep 2/3: Evaluating on test set")
    print("-" * 80)

    from evaluate import main as evaluate_main
    evaluate_main()

    print("\nStep 3/3: Creating visualizations")
    print("-" * 80)

    from visualize import create_visualizations
    create_visualizations()

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nResults saved to:")
    print(f"  - results/test_results.csv")
    print(f"  - results/summary.json")
    print(f"  - results/strategy_comparison.png")
    print(f"  - results/metrics_comparison.png")
    print("\nNext steps:")
    print("  1. Review the visualizations in the results/ directory")
    print("  2. Check the summary.json for performance metrics")
    print("  3. Run 'python main.py --help' for more options")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='RL Options Trading Strategy Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full pipeline (train + eval + viz)
  python main.py --train-only       # Only train the model
  python main.py --eval-only        # Only evaluate (requires trained model)
  python main.py --viz-only         # Only create visualizations
        """
    )

    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only train the model without evaluation'
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only evaluate the trained model (requires existing model)'
    )
    parser.add_argument(
        '--viz-only',
        action='store_true',
        help='Only create visualizations (requires evaluation results)'
    )

    args = parser.parse_args()

    # Check for conflicting flags
    flags = [args.train_only, args.eval_only, args.viz_only]
    if sum(flags) > 1:
        print("Error: Only one of --train-only, --eval-only, or --viz-only can be specified")
        return

    try:
        run_pipeline(
            train_only=args.train_only,
            eval_only=args.eval_only,
            viz_only=args.viz_only
        )
    except Exception as e:
        print(f"\nError running pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
