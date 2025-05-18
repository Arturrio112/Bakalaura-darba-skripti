// USED TO QUERY THE A1111 API TO MAKE VIDEOS WITH CHANGING PARAMETERS

import fetch from 'node-fetch';
import fs from 'fs/promises';
import path from 'path';
// Configuration
const baseUrl = 'http://localhost:7860'; 
const apiEndpoint = '/t2v/run'; 
const baseParams = {
  prompt: "butterfly fly",
  n_prompt: "text, watermark, copyright, blurry, nsfw",
  model: "<modelscope>",
  // sampler: "DDIM_Gaussian",
//   steps: 30,
  // frames: 30,
//   seed: -1,
  cfg_scale: 17,
  // width: 256,
  // height: 256,
  eta: 0,
  batch_count: 1,
  do_vid2vid: false,
  // strength: 0.75,
  vid2vid_startFrame: 0,
  inpainting_frames:0,
  inpainting_weights: '0:(t/max_i_f), "max_i_f":(1)',
  fps: 30,
  add_soundtrack: "None",
  soundtrack_path: "https://deforum.github.io/a1/A1.mp3",
};

// Parameters to iterate through
const seeds = [17, 42, 101];
const stepsValues = [25, 30, 35];
const frames = [30,60,90,120]
const strengths = [0.25,0.5,0.75,1]
const samplers = ["DDIM_Gaussian","DDIM","UniPC"]
const widths = [256,512]
const heigths = [256,512]
// const seeds = [-1];
// const stepsValues = [100];
// const frames = [60]
// const strengths = [1]
// const samplers = ["DDIM_Gaussian"]
// const widths = [256]
// const heigths = [256]

// let skipCount = 1341; // Number of iterations to skip
// let count = 0;

async function getNewestFolder(directory) {
  try {
    const files = await fs.readdir(directory, { withFileTypes: true });

    // Filter only directories
    const folders = files
      .filter(dirent => dirent.isDirectory())
      .map(dirent => path.join(directory, dirent.name));

    if (folders.length === 0) {
      console.error("No folders found in directory.");
      return null;
    }

    // Get modification times for each folder
    let newestFolder = null;
    let newestTime = 0;

    for (const folder of folders) {
      try {
        const stats = await fs.stat(folder);
        if (stats.mtimeMs > newestTime) {
          newestTime = stats.mtimeMs;
          newestFolder = folder;
        }
      } catch (err) {
        console.error(`Error reading folder stats for ${folder}: ${err.message}`);
      }
    }

    return newestFolder;
  } catch (error) {
    console.error(`Error getting newest folder: ${error.message}`);
    return null;
  }
}

async function saveMetadata(jsonData) {
  const baseDir = "E:\\bakis\\stable-diffusion-webui\\outputs\\img2img-images\\text2video\\";

  try {
    const newestFolder = await getNewestFolder(baseDir);
    if (!newestFolder) {
      console.error("Failed to find the newest output folder.");
      return;
    }

    const jsonFilePath = path.join(newestFolder, 'metadata.json');
    await fs.writeFile(jsonFilePath, JSON.stringify(jsonData, null, 2), 'utf-8');
    console.log(`✅ Metadata saved to: ${jsonFilePath}`);
  } catch (error) {
    console.error(`❌ Error saving metadata: ${error.message}`);
  }
}
// Main function to process all parameter combinations
async function processAllCombinations() {
  console.log(`Starting text2video generation with ${seeds.length * stepsValues.length} combinations...`);
  for(const sampler of samplers){
    for(const width of widths){
      for(const heigth of heigths){
        for(const frame of frames){
          for(const strength of strengths){
            for (const seed of seeds) {
              for (const steps of stepsValues) {
                //ONLY WHEN API CRASHES AND NEED TO SKIP FIRST N ITERATIONS
                // count++;

                // if (count <= skipCount) {
                //   console.log(`Skipping iteration ${count}...`);
                //   continue; // Skip first 100 runs
                // }
                console.log(`\nProcessing combination: seed=${seed}, steps=${steps}`);

                const paramsToSave = {
                  ...baseParams,
                  width: width,
                  height: heigth,
                  sampler: sampler,
                  strength:strength,
                  frames: frame,
                  seed: seed,
                  steps: steps
                }

                // Create query parameters
                const params = new URLSearchParams({
                  ...baseParams,
                  width: width.toString(),
                  height: heigth.toString(),
                  sampler: sampler.toString(),
                  strength:strength.toString(),
                  frames: frame.toString(),
                  seed: seed.toString(),
                  steps: steps.toString()
                });

                try {
                  const startTime = Date.now();
                  // Send POST request with query parameters
                  const url = `${baseUrl}${apiEndpoint}?${params.toString()}`;
                  console.log(`Sending POST request to: ${url}`);

                  const response = await fetch(url, {
                    method: 'POST',
                    headers: {
                      'Accept': 'application/json'
                    }
                  });
                  const endTime = Date.now(); // Capture end time
                  const elapsedTime = (endTime - startTime) / 1000
                  if (response.ok) {
                    const data = await response.json();
                    console.log(`✓ Successfully processed with seed=${seed}, steps=${steps}`);
                    console.log(`⏳ Generation time: ${elapsedTime} seconds`);
                    paramsToSave.generation_time = elapsedTime;
                    await saveMetadata(paramsToSave);
                    
                  //   console.log('Response:', data);
                  } else {
                    const errorText = await response.text();
                    console.error(`Error for seed=${seed}, steps=${steps}: ${errorText}`);
                  }

                  // Wait between requests
                  console.log(`Waiting 15 seconds before next request...`);
                  await new Promise(resolve => setTimeout(resolve, 15000));

                } catch (error) {
                  console.error(`Exception for seed=${seed}, steps=${steps}: ${error.message}`);
                }
              }
            }
          }
        }
      }
    }
  }
  console.log("\nAll videos processed!");
}
const start = Date.now();
// Start the process
processAllCombinations()
  .then(() => console.log('Script completed successfully'))
  .catch(err => console.error('Script failed:', err));

const end = Date.now()
const time = (end - start) / 1000
console.log(`⏳ Total run time: ${time} seconds`);
