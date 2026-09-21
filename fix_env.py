"""
Helper script to fix DATABASE_URL in .env file.
Removes 'psql' command prefix and quotes if present.
"""

import os
import re

def fix_database_url():
    """Fix DATABASE_URL in .env file."""
    env_file = '.env'
    
    if not os.path.exists(env_file):
        print(f"Error: {env_file} file not found!")
        return
    
    # Read the file
    with open(env_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process lines
    fixed_lines = []
    found = False
    
    for line in lines:
        if line.strip().startswith('DATABASE_URL='):
            found = True
            # Extract the value part
            value = line.split('=', 1)[1].strip()
            
            # Remove 'psql' prefix if present
            if value.startswith('psql'):
                # Extract the URL from quotes
                match = re.search(r"['\"](postgresql://[^'\"]+)['\"]", value)
                if match:
                    value = match.group(1)
                else:
                    # Try to extract after 'psql'
                    parts = value.split("'", 1)
                    if len(parts) > 1:
                        value = parts[1].rstrip("'")
            
            # Remove quotes if present
            value = value.strip("'\"")
            
            # Ensure it's a valid postgresql URL
            if not value.startswith('postgresql://'):
                print(f"Warning: DATABASE_URL doesn't start with 'postgresql://'")
                print(f"Current value: {value}")
                print("Please manually fix the DATABASE_URL in .env file")
                fixed_lines.append(line)
            else:
                fixed_lines.append(f"DATABASE_URL={value}\n")
                print(f"Fixed DATABASE_URL: {value[:50]}...")
        else:
            fixed_lines.append(line)
    
    if not found:
        print("DATABASE_URL not found in .env file")
        return
    
    # Write back
    with open(env_file, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"\n{env_file} file has been updated!")

if __name__ == '__main__':
    fix_database_url()


