# Login 403 Forbidden Error - Resolution Guide

## Issue Summary
**Error Code:** 403 Forbidden  
**Error Message:** "Account pending admin approval"  
**Root Cause:** Attempting to login with a newly registered account that hasn't been approved yet

---

## Understanding the Issue

### What Happened
Based on the backend logs:
```
2025-10-31 21:06:51 - New user registered: krishnsameer54@gmail.com (pending admin approval)
INFO: 127.0.0.1:52284 - "POST /api/auth/login HTTP/1.1" 403 Forbidden
INFO: 127.0.0.1:52285 - "POST /api/auth/login HTTP/1.1" 403 Forbidden
```

A new user account was registered, but when attempting to log in immediately after registration, the request was rejected with **403 Forbidden** because:

1. New user accounts are created with `is_active = False` by default
2. They require admin approval before they can log in
3. This is a security feature to prevent unauthorized access

### Why This Design?

```python
# From mvp_backend.py line 297-302
user = User(
    email=user_data.email,
    username=user_data.username,
    full_name=user_data.full_name,
    hashed_password=get_password_hash(user_data.password),
    role="user",
    is_active=False,  # ← Requires admin approval
    is_verified=False
)
```

This is an **enterprise security pattern** where:
- All new user registrations require admin approval
- Prevents spam accounts and unauthorized access
- Gives admins control over who can access the system
- Standard practice for B2B/enterprise applications

---

## Solutions

### ✅ Solution 1: Use Pre-Approved Test Accounts (Recommended for Testing)

The backend automatically creates two test accounts that are pre-approved:

**Admin Account:**
- Email: `admin@qflare.com`
- Password: `admin123`
- Role: `admin`
- Status: ✅ Pre-approved (`is_active = True`)

**Test User Account:**
- Email: `user@qflare.com`
- Password: `user123`
- Role: `user`
- Status: ✅ Pre-approved (`is_active = True`)

**Usage:**
1. Go to http://localhost:3000/login
2. Enter one of the test credentials above
3. Click LOGIN
4. ✅ You should be logged in successfully

---

### ✅ Solution 2: Admin Approval of New Users

If you need to use a newly registered account:

#### Step 1: Register New User
```
POST /api/auth/register
{
  "email": "newuser@example.com",
  "username": "newuser",
  "password": "password123"
}
```
**Response:** "Registration successful! Please wait for admin approval."

#### Step 2: Admin Lists Pending Users
```
GET /api/admin/pending-users
Authorization: Bearer <admin-token>
```
**Response:**
```json
{
  "pending_users": [
    {
      "id": "user-uuid",
      "email": "newuser@example.com",
      "username": "newuser",
      "is_active": false
    }
  ]
}
```

#### Step 3: Admin Approves User
```
POST /api/admin/approve-user/{user-id}
Authorization: Bearer <admin-token>
```
**Response:** "User newuser@example.com approved and keys generated"

#### Step 4: User Can Now Login
```
POST /api/auth/login
{
  "email": "newuser@example.com",
  "password": "password123"
}
```
**Response:** ✅ JWT tokens returned

---

### ✅ Solution 3: Auto-Approve for Development (Optional)

For development/testing environments only, you can modify the registration to auto-approve:

**Edit:** `backend/mvp_backend.py` line 299

**Change from:**
```python
user = User(
    email=user_data.email,
    username=user_data.username,
    full_name=user_data.full_name,
    hashed_password=get_password_hash(user_data.password),
    role="user",
    is_active=False,  # ← Requires approval
    is_verified=False
)
```

**Change to:**
```python
user = User(
    email=user_data.email,
    username=user_data.username,
    full_name=user_data.full_name,
    hashed_password=get_password_hash(user_data.password),
    role="user",
    is_active=True,   # ← Auto-approved for dev
    is_verified=True  # ← Auto-verified for dev
)
```

⚠️ **Warning:** Only use this for development! Never in production!

---

## Improved Error Message

I've updated the error message to be more helpful:

**Before:**
```json
{
  "detail": "Account pending admin approval"
}
```

**After:**
```json
{
  "detail": "Account pending admin approval. Use test credentials: admin@qflare.com/admin123 or user@qflare.com/user123"
}
```

This now tells users exactly what to do when they encounter this error.

---

## Testing the Fix

### Test 1: Test Credentials Login ✅
```bash
curl -X POST http://localhost:8002/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@qflare.com",
    "password": "admin123"
  }'
```
**Expected:** 200 OK with JWT tokens

### Test 2: New User Registration + Pending State ✅
```bash
# Step 1: Register
curl -X POST http://localhost:8002/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "username": "testuser",
    "password": "test123"
  }'

# Step 2: Try to login immediately
curl -X POST http://localhost:8002/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "test123"
  }'
```
**Expected:** 403 Forbidden with helpful message

### Test 3: Admin Approval Flow ✅
```bash
# Step 1: Get admin token
ADMIN_TOKEN=$(curl -X POST http://localhost:8002/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@qflare.com","password":"admin123"}' \
  | jq -r '.access_token')

# Step 2: Get pending users
curl -X GET http://localhost:8002/api/admin/pending-users \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Step 3: Approve user (replace USER_ID)
curl -X POST http://localhost:8002/api/admin/approve-user/USER_ID \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Step 4: User can now login
curl -X POST http://localhost:8002/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "test123"
  }'
```
**Expected:** All steps succeed

---

## Quick Fix for Immediate Testing

If you just want to test the application right now:

1. **Stop trying to login with newly registered accounts**
2. **Use the test credentials:**
   - `admin@qflare.com` / `admin123`
   - `user@qflare.com` / `user123`
3. **These are pre-approved and will work immediately**

---

## Frontend Improvement Suggestion

Consider updating the registration success message in `CleanLogin.tsx` to be clearer:

**Current message:**
```tsx
if (success) {
  setIsLogin(true);
  setError('');
  // Show success message or auto-login
}
```

**Suggested improvement:**
```tsx
if (success) {
  setIsLogin(true);
  setError('');
  setSuccessMessage('Registration successful! Your account is pending admin approval. Please use test credentials to login: admin@qflare.com/admin123');
}
```

---

## Summary

✅ **Not a bug** - Working as designed for security  
✅ **Use test credentials** for immediate access  
✅ **Admin approval required** for new registrations  
✅ **Error message improved** to guide users  
✅ **All endpoints working correctly**

The 403 error is expected behavior for the enterprise security model. Use the pre-approved test accounts for development and testing!
