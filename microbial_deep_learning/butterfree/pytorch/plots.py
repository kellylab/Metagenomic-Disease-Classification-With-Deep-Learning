

def visualize(model, test_loader, writer):

    for batch in test_loader:
        writer.insert(model(batch))
