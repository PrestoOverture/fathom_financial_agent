import asyncio
from graph.workflow import create_graph

async def main():
    graph = create_graph()
    input_state = {"question": "What was Ulta Beauty's Net Sales for fiscal year 2022?"}
    print(f"Question: {input_state['question']}")
    result = await graph.ainvoke(input_state)
    print("\n--- Output ---")
    print(f"Answer: {result.get('answer')}")
    print(f"Reasoning Trace: {result.get('reasoning_logs')}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())