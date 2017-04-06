data_path = "data/"
movie_lines_file = "movie_lines.txt"
stripped_lines_file = "stripped_movie_lines.txt"
with open( data_path + movie_lines_file, 'r') as source:
    with open( data_path + stripped_lines_file, 'w' ) as target:
        split_string = "+++$+++"
        for line in source:
            split = line.split( split_string )
            line_id = split[0]
            line_text = split[-1]
            target.write( line_id + split_string + line_text )
