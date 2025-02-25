import torch
import numpy as np
from pymongo import MongoClient

class FaceEmbeddingDB:
    """Class to handle storing and retrieving face embeddings in MongoDB."""
    def __init__(self, client, db_name, collection_name):
        """Initialize MongoDB connection."""
        self.client = MongoClient(client)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.collection2=self.db["stamped_database"]

    def save_embedding(self, name, pos, embedding):
        """Save face embedding to MongoDB."""
        tensor_np = embedding.detach().cpu().numpy()  # Convert tensor to NumPy
        shape = tensor_np.shape  # Store shape for reconstruction
        tensor_bytes = tensor_np.tobytes()  # Convert to bytes for storage

        document = {
            "name": name,
            "pos": pos,
            "tensor": tensor_bytes,
            "shape": shape
        }

        inserted_document = self.collection.insert_one(document)
        print(f"Inserted Row: {inserted_document.inserted_id}")

    def load_embeddings(self, pos):
        """Retrieve stored embeddings based on position."""
        data = list(self.collection.find({"pos": pos}))

        if data:
            for i in data:
                shape = tuple(i["shape"])  # Retrieve stored shape
                retrieved_np = np.frombuffer(i['tensor'], dtype=np.float32).reshape(shape)
                i['tensor'] = torch.tensor(retrieved_np)  # Convert to PyTorch tensor
            return data
        else:
            print(f"No record found for position {pos}")
            return None

    def add_stamp(self,name,date_rn) :
        pos="1F101"
        insp_job="Interior FRLH"
        insp_subjob= "Cabin Fr"
        iDefect ="" 
        sDefect=""
        category=""
        defect_name=""
        inspector=name 
        shift="WHITE"
        dates=date_rn
        document = {
            "pos": pos,
            "insp_job": insp_job,
            "insp_sub_job": insp_subjob,
            "iDefect": iDefect,
            "sDefect": sDefect,
            "category": category,
            "defect_name": defect_name,
            "inspector": inspector,
            "shift": shift,
            "date": dates,

        }
        self.collection2.insert_one(document)
        # print(f"Inserted Row: {inserted_document.inserted_id}")

    def add_data(self, name, pos, embedding):
        """Add a new face embedding for a given position."""
        self.save_embedding(name, pos, embedding)
        print(f"Data added for position {pos}")

    def delete_data(self, pos):
        """Delete all face embeddings for a given position."""
        result = self.collection.delete_many({"pos": pos})
        if result.deleted_count > 0:
            print(f"Deleted {result.deleted_count} record(s) for position {pos}")
        else:
            print(f"No record found for position {pos}")

    def delete_by_name(self, name):
        """Delete face embeddings based on name."""
        result = self.collection.delete_many({"name": name})
        if result.deleted_count > 0:
            print(f"Deleted {result.deleted_count} record(s) for name {name}")
        else:
            print(f"No record found for name {name}")

    def update_data(self, name, new_name=None, new_pos=None, new_embedding=None):
        """Update specified fields (name, position, tensor) of a face embedding based on name."""
        update_fields = {}
        if new_name:
            update_fields["name"] = new_name
        if new_pos:
            update_fields["pos"] = new_pos
        if new_embedding is not None:
            tensor_np = new_embedding.detach().cpu().numpy()
            update_fields["tensor"] = tensor_np.tobytes()
            update_fields["shape"] = tensor_np.shape
        
        if update_fields:
            result = self.collection.update_many({"name": name}, {"$set": update_fields})
            if result.modified_count > 0:
                print(f"Updated {result.modified_count} record(s) for name {name}")
            else:
                print(f"No record found for name {name}")
        else:
            print("No fields provided for update.")        

    def close(self):
        """Close MongoDB connection."""
        self.client.close()
