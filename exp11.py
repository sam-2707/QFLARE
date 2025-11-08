import math
from math import gcd
def extended_gcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = extended_gcd(b % a, a)
        return (g, x - (b // a) * y, y)
def modInv(e, phi):
    g, x, y = extended_gcd(e, phi)
    if g != 1:
        return None # inverse doesn't exist
    else:
        return x % phi
def isPrime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True
def encrypt(msg, e, n):
    encrypted = []
    for ch in msg:
        m = ord(ch)
        if m >= n:
            return []
        c = pow(m, e, n)
        encrypted.append(c)
    return encrypted
def decrypt(ciphertext, d, n):
    decrypted = []
    for x in ciphertext:
        m = pow(x, d, n)
        decrypted.append(chr(m))
    return ''.join(decrypted)
p = None
q = None
e = None
n = None
d = None
has_keys = False
while True:
    print("\n=== RSA Cipher System ===")
    print("1. Key Generation\n2. Encrypt\n3. Decrypt\n4. Exit\nChoice: ")
    ch = input().strip()
    if ch == "4":
        break
    try:
        ch = int(ch)
        if ch == 1:
            p = int(input("Enter prime p: "))
            q = int(input("Enter prime q: "))
            if not isPrime(p) or not isPrime(q) or p == q:
                print("Invalid primes!\n")
                continue
            n = p * q
            phi = (p - 1) * (q - 1)
            e = int(input("Enter public exponent e: "))
            if math.gcd(e, phi) != 1:
                print("e is not coprime to φ!\n")
                continue
            d = modInv(e, phi)
            if d is None:
                print("No inverse exists for e!\n")
                continue
            print(f"\nPublic key: (e={e}, n={n})\nPrivate key: (d={d}, n={n})")
            if n < 128:
                print("n too small for ASCII!")
            has_keys = True
        elif ch == 2 and has_keys:
            msg = input("\nEnter plaintext: ")
            ciphertext = encrypt(msg, e, n)
            if not ciphertext:
                continue
            print("Cipher:", ' '.join(map(str, ciphertext)))
        elif ch == 3 and has_keys:
            cnt = int(input("Ciphertext count: "))
            ciphertext = list(map(int, input().split()))
            if len(ciphertext) != cnt:
                print("Invalid ciphertext count!\n")
                continue
            plaintext = decrypt(ciphertext, d, n)
            print(f"Plain: {plaintext}")
        else:
            print("Generate keys first!\n")
    except ValueError:
        print("\nPlease enter a valid integer!\n")