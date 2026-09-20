import argparse


def ingest(config_path: str, rebuild: bool) -> None:
    from process_document import build_index
    from rag import load_config

    config = load_config(config_path)
    index_config = config.get("index", {})
    model_config = config.get("models", {})
    build_index(
        config["documents"],
        db_name=index_config.get("persist_directory", "my_chroma_db"),
        embedding_model_name=model_config.get(
            "embedding", "BAAI/bge-base-en-v1.5"
        ),
        rebuild=rebuild,
    )


def ask_once(query: str, config_path: str) -> None:
    from ask_rag import ask
    from rag import create_pipeline

    ask(query, pipeline=create_pipeline(config_path))


def chat(config_path: str) -> None:
    from ask_rag import ask
    from rag import create_pipeline

    pipeline = create_pipeline(config_path)
    while True:
        query = input("Enter your question (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break
        if query:
            ask(query, pipeline=pipeline)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local research-paper RAG")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser("ingest", help="build the document index")
    ingest_parser.add_argument("--config", default="config.json")
    ingest_parser.add_argument(
        "--rebuild", action="store_true", help="replace an existing index"
    )

    ask_parser = commands.add_parser("ask", help="ask one question")
    ask_parser.add_argument("query")
    ask_parser.add_argument("--config", default="config.json")

    chat_parser = commands.add_parser(
        "chat", help="start an interactive question loop"
    )
    chat_parser.add_argument("--config", default="config.json")

    evaluate_parser = commands.add_parser(
        "evaluate", help="run the configured evaluation"
    )
    evaluate_parser.add_argument("--config", default="config.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "ingest":
        ingest(args.config, args.rebuild)
    elif args.command == "ask":
        ask_once(args.query, args.config)
    elif args.command == "chat":
        chat(args.config)
    elif args.command == "evaluate":
        from evaluation import evaluate
        evaluate(args.config)


if __name__ == "__main__":
    main()
