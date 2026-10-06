import os
SEED = int(os.getenv("SEED", 34))

def main():

    print(f"Using seed: {SEED}")

    return True

if __name__ == "__main__":
    main()
