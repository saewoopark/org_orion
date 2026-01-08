#!/usr/bin/env python3

import subprocess

def main():
    filenames = [
        "resnet15_11.yml", "resnet15_13.yml", "resnet15_15.yml",
        "resnet15_6.yml",  "resnet15_8.yml",
        "resnet16_10.yml", 
        "resnet16_12.yml", "resnet16_14.yml",
        "resnet16_16.yml", "resnet16_7.yml", 
        "resnet16_9.yml",
        "resnet15_10.yml", 
        "resnet15_12.yml", "resnet15_14.yml",
        "resnet15_16.yml", "resnet15_7.yml",  "resnet15_9.yml",
        "resnet16_11.yml", "resnet16_13.yml", "resnet16_15.yml",
        "resnet16_6.yml",  "resnet16_8.yml",
    ]

    for filename in filenames:
        output_file = f"{filename}_latency_nexus1.txt"
        command = ["python3", "run_resnet_latency.py", "/home/japark/env_orion/orion/configs/"+filename]

        print(f"Running: {' '.join(command)} > {output_file}")

        with open(output_file, "w") as f:
            subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)

if __name__ == "__main__":
    main()
