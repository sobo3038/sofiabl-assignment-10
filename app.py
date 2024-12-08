from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import torch
import pandas as pd
from PIL import Image
from open_clip import create_model_and_transforms, tokenizer
import torch.nn.functional as F
from sklearn.decomposition import PCA

app = Flask(__name__)


model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')
model.eval()
df = pd.read_pickle('image_embeddings.pickle')
embeddings = torch.tensor([row['embedding'] for _, row in df.iterrows()])
UPLOAD_FOLDER = 'uploads'
IMAGE_FOLDER = 'coco_images_resized'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



@app.route('/coco_images_resized/<path:filename>')
def serve_images(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)


def search(query_embedding, top_k=5, use_pca=False, n_components=5):
    """
    Perform a search for the most similar images based on embeddings.
    Optionally applies PCA for dimensionality reduction.
    """
    if use_pca:
        embeddings_np = embeddings.detach().numpy()
        query_np = query_embedding.detach().numpy()

        pca = PCA(n_components=min(n_components, embeddings_np.shape[1]))
        reduced_embeddings = pca.fit_transform(embeddings_np)
        reduced_query = pca.transform(query_np)

        reduced_embeddings = torch.tensor(reduced_embeddings)
        reduced_query = torch.tensor(reduced_query)
        cos_similarities = torch.mm(reduced_query, reduced_embeddings.T)
    else:
        cos_similarities = torch.mm(query_embedding, embeddings.T)

    top_k_indices = torch.topk(cos_similarities, top_k, dim=1).indices[0]
    results = []
    for idx in top_k_indices:
        file_name = df.iloc[idx.item()]['file_name']
        score = cos_similarities[0, idx].item()
        results.append((os.path.join("/coco_images_resized", file_name), score))
    return results

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main route for the web application.
    Handles text, image, and hybrid queries with optional PCA.
    """
    results = []
    if request.method == "POST":
        query_type = request.form["query_type"]
        hybrid_weight = float(request.form.get("hybrid_weight", 0.5))
        text_query = request.form.get("text_query", "").strip()
        use_pca = "use_pca" in request.form
        n_components = int(request.form.get("n_components", 5))
        image_query_path = None

        if "image_query" in request.files:
            file = request.files["image_query"]
            if file:
                filename = secure_filename(file.filename)
                image_query_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(image_query_path)
        
        if query_type == "text_query" and text_query:
            text_tokens = tokenizer.tokenize([text_query]).to("cpu")
            text_query_embedding = F.normalize(model.encode_text(text_tokens))
            results = search(text_query_embedding, use_pca=use_pca, n_components=n_components)

        elif query_type == "image_query" and image_query_path:
            image = preprocess(Image.open(image_query_path)).unsqueeze(0)
            image_query_embedding = F.normalize(model.encode_image(image))
            results = search(image_query_embedding, use_pca=use_pca, n_components=n_components)

        elif query_type == "hybrid_query" and image_query_path and text_query:
            text_tokens = tokenizer.tokenize([text_query]).to("cpu")
            text_query_embedding = F.normalize(model.encode_text(text_tokens))
            image = preprocess(Image.open(image_query_path)).unsqueeze(0)
            image_query_embedding = F.normalize(model.encode_image(image))
            hybrid_query_embedding = F.normalize(
                hybrid_weight * text_query_embedding + (1.0 - hybrid_weight) * image_query_embedding
            )
            results = search(hybrid_query_embedding, use_pca=use_pca, n_components=n_components)

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
