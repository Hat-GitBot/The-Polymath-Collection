# 🚀 Streamlit Cloud Deployment Guide

## Step-by-Step Instructions

### 1. Prepare Your Files

Ensure you have these files in your repository:
- ✅ `main_app.py` (portfolio landing page)
- ✅ `pages/` folder with all 5 page files
- ✅ `requirements.txt`
- ✅ `nltk.txt`
- ✅ `packages.txt`
- ✅ `.streamlit/config.toml`
- ✅ `README.md`

### 2. Create GitHub Repository
```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: The Polymath Collection"

# Create repository on GitHub, then:
git remote add origin https://github.com/yourusername/polymath-collection.git
git branch -M main
git push -u origin main
```

### 3. Deploy to Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository: `yourusername/polymath-collection`
5. Set branch: `main`
6. Set main file path: `main_app.py`
7. Click **"Deploy"**

### 4. Wait for Deployment

- ⏱️ Initial deployment: 3-5 minutes
- 📦 Installing dependencies from requirements.txt
- 📚 Downloading NLTK data from nltk.txt
- 🎨 Applying theme from .streamlit/config.toml

### 5. Your App is Live! 🎉

Your portfolio will be available at:
```
https://your-app-name.streamlit.app
```

## Troubleshooting

### Issue: NLTK Data Not Found

**Solution:** Ensure `nltk.txt` is in root directory with correct package names.

### Issue: Module Not Found

**Solution:** Check all dependencies are in `requirements.txt` with correct versions.

### Issue: App Crashes on Startup

**Solution:** Check Streamlit Cloud logs for specific error messages.

### Issue: Upload Size Limit

**Solution:** Streamlit Cloud has 200MB upload limit (already configured in config.toml).

## Custom Domain (Optional)

To use a custom domain:
1. Go to app settings in Streamlit Cloud
2. Click "Custom domain"
3. Follow instructions to configure DNS

## Environment Variables (If Needed)

If you need API keys:
1. Go to app settings
2. Click "Secrets"
3. Add in TOML format:
```toml
   api_key = "your-key-here"
```

## Updating Your App
```bash
# Make changes locally
git add .
git commit -m "Update: description of changes"
git push

# Streamlit Cloud will auto-deploy within 1-2 minutes
```

## Monitoring

- View app logs in Streamlit Cloud dashboard
- Monitor resource usage
- Check deployment history

---

**Need help?** Check [Streamlit Community](https://discuss.streamlit.io/)
