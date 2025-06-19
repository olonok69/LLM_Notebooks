# Managing Multiple GitHub Accounts

This guide shows you how to work with multiple GitHub accounts on the same machine using different methods.

## Method 1: SSH Keys (Recommended)

### Step 1: Generate SSH Keys

Create separate SSH keys for each account:

```bash
# Generate key for work account
ssh-keygen -t ed25519 -C "work@company.com" -f ~/.ssh/id_ed25519_work

# Generate key for personal account
ssh-keygen -t ed25519 -C "personal@email.com" -f ~/.ssh/id_ed25519_personal
```

### Step 2: Add Keys to SSH Agent

```bash
# Start SSH agent
eval "$(ssh-agent -s)"

# Add both keys
ssh-add ~/.ssh/id_ed25519_work
ssh-add ~/.ssh/id_ed25519_personal
```

### Step 3: Configure SSH Config

Create or edit `~/.ssh/config`:

```
# Work GitHub account
Host github-work
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_work
    IdentitiesOnly yes

# Personal GitHub account  
Host github-personal
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_personal
    IdentitiesOnly yes
```

### Step 4: Add Public Keys to GitHub

Copy and add public keys to respective GitHub accounts:

```bash
# Copy work public key
cat ~/.ssh/id_ed25519_work.pub

# Copy personal public key
cat ~/.ssh/id_ed25519_personal.pub
```

Go to GitHub Settings → SSH and GPG keys → New SSH key for each account.

### Step 5: Clone Repositories

Use the custom hostnames when cloning:

```bash
# Clone work repository
git clone git@github-work:company/work-repo.git

# Clone personal repository
git clone git@github-personal:username/personal-repo.git
```

### Step 6: Configure Git Identity Per Repository

Set the correct user info for each repository:

```bash
# In work repository
cd work-repo
git config user.name "Your Work Name"
git config user.email "work@company.com"

# In personal repository
cd personal-repo
git config user.name "Your Personal Name"
git config user.email "personal@email.com"
```

## Method 2: Directory-Based Configuration

### Step 1: Create Directory Structure

Organize your repositories:

```
~/
├── work/
│   └── project1/
│   └── project2/
└── personal/
    └── project1/
    └── project2/
```

### Step 2: Configure Global Git Config

Edit `~/.gitconfig`:

```ini
[user]
    name = Your Default Name
    email = default@email.com

[includeIf "gitdir:~/work/"]
    path = ~/.gitconfig-work

[includeIf "gitdir:~/personal/"]
    path = ~/.gitconfig-personal
```

### Step 3: Create Separate Config Files

Create `~/.gitconfig-work`:

```ini
[user]
    name = Your Work Name
    email = work@company.com
```

Create `~/.gitconfig-personal`:

```ini
[user]
    name = Your Personal Name
    email = personal@email.com
```

## Method 3: Personal Access Tokens

### Step 1: Generate Tokens

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate separate tokens for each account
3. Save them securely

### Step 2: Clone with Tokens

```bash
# Clone with work token
git clone https://work-token@github.com/company/repo.git

# Clone with personal token
git clone https://personal-token@github.com/username/repo.git
```

## Quick Commands Reference

### Check Current Configuration

```bash
# Check current user config
git config user.name
git config user.email

# Check SSH connection
ssh -T git@github-work
ssh -T git@github-personal
```

### Switch Remote URL

If you need to change the remote URL of an existing repository:

```bash
# Change to work account
git remote set-url origin git@github-work:company/repo.git

# Change to personal account
git remote set-url origin git@github-personal:username/repo.git
```

### Verify Setup

Test your setup:

```bash
# Test SSH connections
ssh -T git@github-work
ssh -T git@github-personal

# Check which key is being used
ssh -vT git@github-work
```

## Troubleshooting

### Permission Denied

If you get permission denied errors:

1. Check if SSH agent is running: `ssh-add -l`
2. Ensure correct SSH config syntax
3. Verify public keys are added to correct GitHub accounts
4. Test SSH connection: `ssh -T git@github-work`

### Wrong Account Used

If commits show wrong author:

1. Check local git config: `git config --list`
2. Set correct user info: `git config user.email "correct@email.com"`
3. Amend last commit if needed: `git commit --amend --reset-author`

### Multiple SSH Keys Conflict

If you have issues with multiple keys:

1. Add `IdentitiesOnly yes` to SSH config
2. Clear SSH agent: `ssh-add -D`
3. Re-add specific keys: `ssh-add ~/.ssh/id_ed25519_work`

## Best Practices

- Use descriptive SSH key names and comments
- Keep work and personal repositories in separate directories
- Always verify your git config before making commits
- Use SSH keys over HTTPS for better security
- Regularly rotate personal access tokens
- Test your setup after configuration changes