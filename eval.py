import numpy as np
import os
import pandas as pd
import torch
from sklearn.metrics import f1_score
import argparse
from dataset import get_dataloader,EmotionDataset
from torch.utils.data import Dataset,DataLoader


class EmotiontestDataset(Dataset):
    def __init__(self, data):
        self.id = [id for id in data['id']]
        self.features = [torch.FloatTensor(feature) for feature in data['feature']]

    def __getitem__(self, idx):
        return self.id[idx],self.features[idx]
        
    def __len__(self):
        return len(self.features)


def load_data(args):
    feature_path = os.path.join(args.dataset_file_path, args.feature_set)
    data = {'id':[],"feature":[]}
       
    for vid in os.listdir(feature_path):
        vid_path = os.path.join(feature_path, vid)
        if os.path.isdir(vid_path):
            for file in sorted(os.listdir(vid_path)):
                if file.endswith("csv"):
                    feature = pd.read_csv(os.path.join(vid_path, file), header=None).to_numpy().astype(np.float32)
                elif file.endswith("npy"):
                    feature = np.load(os.path.join(vid_path, file)).astype(np.float32)
                
                if feature.ndim == 2:
                    feature = np.max(feature, axis=0, keepdims=True)
                data["id"].append(vid)
                data["feature"].append(feature)

    return data

def evaluate(model, data_loader):
    model.eval()
    all_predictions = []
    all_id = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            id, features = batch_data
            features = features.cuda()
            
            predictions = model(features)
            all_id.extend([i for i in id])  
            all_predictions.append(predictions.cpu().numpy())

    all_id = np.array(all_id).reshape(-1, 1)  
    all_predictions = np.vstack(all_predictions)
    
    return all_id, all_predictions

# Take the average of the sample based on the subject ID
def save_predictions(id, predictions, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.DataFrame({
        'id': id.flatten(),  
            'Extraversion': predictions[:, 0].astype(float),
        'Agreeableness': predictions[:, 1].astype(float),
        'Conscientiousnes': predictions[:, 2].astype(float),
        'Neuroticism': predictions[:, 3].astype(float),
        'Open Mindedness': predictions[:, 4].astype(float)
    })

    df['id'] = df['id'].astype(str)

    df_grouped = df.groupby('id', as_index=False).agg({
        'Extraversion': 'mean',
        'Agreeableness': 'mean',
        'Conscientiousnes': 'mean',
        'Neuroticism': 'mean',
        'Open Mindedness': 'mean'
    })
    
    df_grouped.to_csv(save_path, index=False)
    print(f'The predicted results have been saved to: {save_path}')




def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_file_path', default="xxx/MER2025_personality/deception/features/")
    parser.add_argument('--data_source', default="deception")
    parser.add_argument('--feature_set', default="baichuan13B-base", type=str)
    parser.add_argument('--fea_dim', default=768, type=int)
    

    parser.add_argument('--model_path',default="./model/2025-04-23-11-20-17_[deception_baichuan13B-base]_[1e-05_512_0.5_64]/36.pth")
    parser.add_argument('--output_dir', default='./predictions/predictions2.csv')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--classnum', default=5, type=int)
    
    args = parser.parse_args()
    
    print('Load data...')
    data = load_data(args)

    test_dataset = EmotiontestDataset(data)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)


    print(f'Load model from: {args.model_path}')
    model = torch.load(args.model_path, weights_only=False)
    model.cuda()
    model.eval()
    
    print('Save predictions...')
    id, predictions = evaluate(model, test_dataloader)
    
    save_predictions(id, predictions, args.output_dir)
    



if __name__ == '__main__':
    main()