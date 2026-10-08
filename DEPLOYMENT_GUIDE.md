# Deployment Guide for ChurnGuard

This guide walks you through deploying the **Customer Churn Prediction System (ChurnGuard)** to production. All necessary deployment configuration files (`Dockerfile`, `.dockerignore`, `.streamlit/config.toml`, `render.yaml`, `requirements.txt`) are already configured.

---

## 🌟 Method 1: Streamlit Community Cloud (Recommended — 100% Free)

Streamlit Community Cloud provides free, instant cloud hosting with continuous deployment from GitHub and an automatic HTTPS URL.

### Step 1: Push Project to GitHub

1. Open your browser and go to [GitHub.com/new](https://github.com/new).
2. Create a new repository named `customer-churn-prediction` (set it to **Public**).
3. In PowerShell, link your local repository and push:
   ```powershell
   cd C:\Users\vaishnavi\.gemini\antigravity\scratch\customer-churn-prediction
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/customer-churn-prediction.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
2. Click the **"New app"** button.
3. Fill in the deployment details:
   - **Repository:** `<YOUR_GITHUB_USERNAME>/customer-churn-prediction`
   - **Branch:** `main`
   - **Main file path:** `app/main.py`
   - **App URL (optional):** Choose a custom subdomain (e.g. `churnguard-ai.streamlit.app`)
4. Click **"Deploy!"**
5. Within 1–2 minutes, your application will be live at `https://churnguard-ai.streamlit.app`!

---

## 🤗 Method 2: Hugging Face Spaces (Free ML Hosting)

Hugging Face Spaces offers free cloud hosting specifically tailored for Machine Learning apps.

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and log in.
2. Click **"Create new Space"**.
3. Choose:
   - **Space name:** `churnguard-ai`
   - **Space SDK:** `Streamlit`
   - **Space hardware:** Free CPU tier
4. Hugging Face will provide a git clone URL. Push your code:
   ```powershell
   git remote add hf https://huggingface.co/spaces/<YOUR_HF_USERNAME>/churnguard-ai
   git push hf main
   ```
5. Your application will build and launch automatically.

---

## 🚀 Method 3: Render.com (Free Web Service)

Render allows free cloud hosting of Python web apps directly from GitHub using the included `render.yaml`.

1. Go to [dashboard.render.com](https://dashboard.render.com/) and connect your GitHub account.
2. Click **"New +"** $\rightarrow$ **"Web Service"**.
3. Select your `customer-churn-prediction` repository.
4. Set:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app/main.py --server.port $PORT --server.address 0.0.0.0`
5. Click **"Deploy Web Service"**.

---

## 🐳 Method 4: Docker Container Deployment

You can run the application inside an isolated Docker container on your local machine or on cloud VMs (AWS EC2, Google Cloud Compute, DigitalOcean, Azure):

### Build and Run with Docker:
```powershell
# Navigate to the project directory
cd C:\Users\vaishnavi\.gemini\antigravity\scratch\customer-churn-prediction

# Build the Docker image
docker build -t churnguard-ai .

# Run the container
docker run -d -p 8501:8501 --name churnguard-app churnguard-ai
```
Access the application at `http://localhost:8501`.

---

## ⚡ Method 5: Instant Public URL for Testing (No Cloud Setup Needed)

If you need an immediate public link to demonstrate to teachers or peers right now without pushing to GitHub first:

You can use **localtunnel** or **Cloudflare Tunnel**:

### Using Cloudflare Tunnel:
```powershell
# Download cloudflared or run if installed:
cloudflared tunnel --url http://localhost:8501
```

### Using LocalTunnel (via Node.js/npx):
```powershell
npx localtunnel --port 8501
```
This generates a temporary public HTTPS URL (e.g. `https://quiet-river-12.loca.lt`) accessible from any phone or computer.

---

## 🔑 Demo Account Credentials for Deployed App
Once deployed, reviewers can log in with:
- **Email:** `demo@churnguard.ai`
- **Password:** `Admin@123`
*(Or click the **"⚡ Use Demo Account"** button on the landing page, or register a new personal account).*
