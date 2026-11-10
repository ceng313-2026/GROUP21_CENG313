import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device, mel_transform, amplitude_to_db):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for waveforms, labels in pbar:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        # Cihaz üzerinde (GPU) spektrogram dönüşümü
        mels = mel_transform(waveforms)
        mels_db = amplitude_to_db(mels)

        optimizer.zero_grad()
        outputs = model(mels_db)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f"{running_loss/len(dataloader):.4f}", 'acc': f"{correct/total:.4f}"})

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc
    
    
    
def evaluate(model, dataloader, criterion, device, mel_transform, amplitude_to_db):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for waveforms, labels in pbar:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            
            # Cihaz üzerinde spektrogram dönüşümü
            mels = mel_transform(waveforms)
            mels_db = amplitude_to_db(mels)
            
            outputs = model(mels_db)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
            pbar.set_postfix({'val_loss': f"{running_loss/len(dataloader):.4f}", 'val_acc': f"{correct/total:.4f}"})
            
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc