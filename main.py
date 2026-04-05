import json

def main():
    config= json.load(open('config.json'))
    eval= config.get("eval")
    if eval:
        from evaluation import evaluate
        evaluate()        
    else:
        from ask_rag import ask
        while True:
            query= input("Enter your question (or 'exit' to quit): ")
            if query.lower()=='exit':
                break

            ask(query)



if __name__ == "__main__":
    main()
