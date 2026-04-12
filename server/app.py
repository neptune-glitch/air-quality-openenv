from inference import app
import os

def main():
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, use_reloader=False)

if __name__ == "__main__":
    main()