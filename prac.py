import asyncio

# async def work(n):
#     print(f"Start {n}")
#     await asyncio.sleep(n)
#     print(f"End {n}")
#     return n

# async def main():
#     results = await asyncio.gather(work(1), work(2), work(3))
#     print(f"Results: {results}")

# asyncio.run(main())
# print("judgement")

async def foo():
    await asyncio.sleep(1)
    return "foo"

async def bar():
    await asyncio.sleep(2)
    return "bar"

async def main():
    t1 = asyncio.create_task(foo())
    t2 = asyncio.create_task(bar())

    print("Tasks scheduled, doing something else...")
    result1 = await t1
    result2 = await t2
    print(result1, result2)

asyncio.run(main())
