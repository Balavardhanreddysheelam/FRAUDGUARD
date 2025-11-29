# ⚠️ SECURITY WARNING - Kaggle API Token

## 🚨 CRITICAL: Remove Hardcoded Token

The training notebook (`FraudGuard_v2_Training.ipynb`) contains a **hardcoded Kaggle API token** that must be removed before committing to version control.

### Current Issue
- **Location**: Cell 6 in the notebook
- **Token**: `KGAT_448d97d3de442e268f6bcf6386409cf6`
- **Risk**: If this token is committed to a public repository, it can be used by anyone

### ✅ Solution

**Option 1: Use getpass (Recommended)**
```python
import getpass

KAGGLE_USERNAME = input("Enter your Kaggle username: ").strip()
KAGGLE_API_TOKEN = getpass.getpass("Enter your Kaggle API token: ").strip()
```

**Option 2: Upload kaggle.json file (Best for Colab)**
```python
from google.colab import files
uploaded = files.upload()  # Upload kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

**Option 3: Use environment variables**
```python
import os
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")
```

### 🔒 Action Required

1. **If you've already committed the notebook with the token:**
   - Revoke the token immediately at: https://www.kaggle.com/account
   - Generate a new token
   - Remove the token from git history (if needed)

2. **Before committing:**
   - Remove the hardcoded token from the notebook
   - Use one of the secure methods above
   - Verify `.gitignore` includes `*.ipynb` or the notebook is in a private repo

3. **Best Practice:**
   - Never commit API tokens or secrets
   - Use environment variables or secure input methods
   - Add tokens to `.gitignore` if stored locally

### 📝 How to Get Your Kaggle API Token

1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New Token"
4. This downloads `kaggle.json` with your credentials
5. **Never commit this file to version control!**

### ✅ Verification

After fixing, verify:
- [ ] No hardcoded tokens in the notebook
- [ ] `.gitignore` includes `kaggle.json` and `.env`
- [ ] Token is not in git history (check with `git log -p`)
- [ ] If token was exposed, it's been revoked and regenerated

