import os
import json
BASE_PATH = r"E:\bakis\stable-diffusion-webui\outputs\img2img-images\text2video"
END_PATH = r"E:\bakis\pipelineScripts"
end_json = []


def main():
    try:
        folders = [os.path.join(BASE_PATH, f) for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))]
        print(f"Found {len(folders)} folders to process")
        
        for folder in folders:
            # Load metrics.json
            metrics_path = os.path.join(folder, "metrics.json")
            with open(metrics_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            end_json.append(data)

        # Save results
        end_path = os.path.join(END_PATH, "all.json")
        with open(end_path, 'w', encoding='utf-8') as f:
            json.dump(end_json, f, indent=2)
        print(f"✔ Saved all metrics to {end_path}")
            
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()