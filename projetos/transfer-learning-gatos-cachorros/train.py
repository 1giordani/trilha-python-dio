"""Alternativa ao Colab: execute na pasta do projeto com as dependências instaladas."""
import argparse
from pipeline import (Config, setup, load_data, show_samples, build_model,
                      train_stages, plot_history, evaluate_and_export)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--fine-epochs", type=int, default=3)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--evaluation-limit", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--output-dir", default="resultados")
    args = parser.parse_args()
    if args.head_epochs < 1 or args.fine_epochs < 1 or min(args.train_limit, args.evaluation_limit) < 0 or args.image_size < 32:
        parser.error("Use épocas positivas e limite não negativo.")
    config = Config(head_epochs=args.head_epochs, fine_epochs=args.fine_epochs,
                    train_limit=args.train_limit, evaluation_limit=args.evaluation_limit,
                    image_size=args.image_size, output_dir=args.output_dir)
    setup(config)
    (train, validation, test), metadata = load_data(config)
    show_samples(train, metadata["classes"], config.output_dir)
    model, base = build_model(config)
    model.summary()
    histories, comparison = train_stages(model, base, train, validation, config)
    plot_history(histories, config.output_dir)
    evaluate_and_export(model, test, metadata, histories, comparison, config)


if __name__ == "__main__":
    main()
