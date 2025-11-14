if __name__ == "__main__":
    file_name = ['Abortion', 'Brexit', 'Election', 'Gun', 'Partisanship', 'Referendum']
    for name in file_name:
        input_path = f'./{name}/edgelist_pads'
        output_path = f'./{name}/{name}.txt'

        # read in the file, where each line contains 7 numbers, keep only 0th, 3rd, 6th
        # the first line has only two numbers, keep both of them
        with open(input_path, 'r') as f:
            lines = f.readlines()
        with open(output_path, 'w') as f:
            f.write(lines[0])  # write the first line as is
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) == 7:
                    f.write(f"{parts[0]} {parts[3]} {parts[6]}\n")