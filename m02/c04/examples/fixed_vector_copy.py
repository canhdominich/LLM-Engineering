from _copy_task import evaluate, train

for hidden in (32, 128):
    model, loss = train(hidden=hidden, attention=False)
    params = sum(p.numel() for p in model.parameters())
    print(f"no attention, context vector = {hidden} numbers, params = {params:,}, last loss = {loss:.3f}")
    print("  length  digit accuracy  whole code right")
    for n in (5, 10, 15, 20):
        digit, whole = evaluate(model, n)
        print(f"  {n:6d}  {digit:14.3f}  {whole:16.3f}")
