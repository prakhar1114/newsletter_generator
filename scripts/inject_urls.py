from summariser.ingest import ingest_all_from_urls_path
import asyncio

async def main():
    results = await ingest_all_from_urls_path(workers=4)
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
