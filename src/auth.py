"""
User Authentication and Database Management Module
Author: ChurnGuard AI Team
Description:
    Provides SQLite user authentication with PBKDF2 password hashing + salts,
    input validation, profile management, and session control.
"""

import sqlite3
import hashlib
import os
import re

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database")
DB_PATH = os.path.join(DB_DIR, "users.db")

def get_connection():
    """Get database connection with row factory."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str, salt: str = None) -> tuple:
    """Hash password using PBKDF2-HMAC-SHA256 with cryptographic salt."""
    if salt is None:
        salt = os.urandom(16).hex()
    pwd_bytes = password.encode('utf-8')
    salt_bytes = salt.encode('utf-8')
    key = hashlib.pbkdf2_hmac('sha256', pwd_bytes, salt_bytes, 100000)
    return key.hex(), salt

def verify_password(stored_hash: str, salt: str, provided_password: str) -> bool:
    """Verify input password against stored PBKDF2 hash."""
    hashed, _ = hash_password(provided_password, salt)
    return hashed == stored_hash

def init_db():
    """Initialize users table and insert demo user if not present."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            organization TEXT NOT NULL,
            role TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    
    # Check if demo account exists
    cursor.execute("SELECT id FROM users WHERE email = ?", ("demo@churnguard.ai",))
    if not cursor.fetchone():
        demo_hash, demo_salt = hash_password("Admin@123")
        cursor.execute("""
            INSERT INTO users (full_name, email, organization, role, password_hash, salt)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "Alex Morgan",
            "demo@churnguard.ai",
            "Global Telecom Enterprise",
            "Lead Retention Strategist",
            demo_hash,
            demo_salt
        ))
        conn.commit()
    conn.close()

def is_valid_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email.strip()))

def register_user(full_name: str, email: str, organization: str, role: str, password: str, confirm_password: str) -> tuple:
    """
    Validates credentials and registers new user.
    Returns (success: bool, message: str)
    """
    full_name = full_name.strip()
    email = email.strip().lower()
    organization = organization.strip()
    role = role.strip()
    
    if not full_name:
        return False, "Full Name is required."
    if not is_valid_email(email):
        return False, "Please enter a valid email address."
    if not organization:
        return False, "Organization name is required."
    if not role:
        return False, "Professional role is required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    if password != confirm_password:
        return False, "Passwords do not match."
        
    pwd_hash, salt = hash_password(password)
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (full_name, email, organization, role, password_hash, salt)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (full_name, email, organization, role, pwd_hash, salt))
        conn.commit()
        conn.close()
        return True, "Account registered successfully! You can now log in."
    except sqlite3.IntegrityError:
        return False, f"An account with email '{email}' already exists."
    except Exception as e:
        return False, f"Registration error: {str(e)}"

def authenticate_user(email: str, password: str) -> tuple:
    """
    Authenticates user credentials.
    Returns (user_dict or None, message: str)
    """
    email = email.strip().lower()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None, "Invalid email or password."
        
    stored_hash = row["password_hash"]
    salt = row["salt"]
    
    if verify_password(stored_hash, salt, password):
        user = {
            "id": row["id"],
            "full_name": row["full_name"],
            "email": row["email"],
            "organization": row["organization"],
            "role": row["role"],
            "created_at": row["created_at"]
        }
        return user, "Authentication successful."
    else:
        return None, "Invalid email or password."

def update_user_profile(user_id: int, full_name: str, organization: str, role: str) -> tuple:
    """Updates profile attributes for logged in user."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users
            SET full_name = ?, organization = ?, role = ?
            WHERE id = ?
        """, (full_name.strip(), organization.strip(), role.strip(), user_id))
        conn.commit()
        conn.close()
        return True, "Profile updated successfully."
    except Exception as e:
        return False, f"Failed to update profile: {str(e)}"

def get_user_by_id(user_id: int):
    """Retrieve user dictionary by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, full_name, email, organization, role, created_at FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at", DB_PATH)
    # Test authentication with demo account
    user, msg = authenticate_user("demo@churnguard.ai", "Admin@123")
    print("Demo user authentication test:", "SUCCESS" if user else "FAILED", msg)
