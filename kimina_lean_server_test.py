from kimina_client import KiminaClient

client = KiminaClient("http://localhost:80")
client.run_benchmark(dataset_name="Goedel-LM/Lean-workbook-proofs", 
                     n=1000,
                     batch_size=8,
                     max_workers=4)  # Reduced to avoid overwhelming server