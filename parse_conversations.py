import os
from tensorflow.python.platform import gfile

def get_movie_lines_map():
	split_string = "+++$+++"
	movie_lines_file = os.path.join("data/","movie_lines.txt")
	line_map = dict()
	with gfile.GFile(movie_lines_file, mode = "rb") as file:
		for line in file:
			split = line.split(split_string)
			line_map[ split[0].strip() ] = split[4] #maps line number to the words of the line


	return( line_map )

if __name__ == '__main__':
	global split_string
	split_string = "+++$+++"
	movie_conversations = os.path.join("data/", "movie_conversations.txt")
	output_path = os.path.join("data/", "movie_conversations_actual_text.txt")
	line_map = get_movie_lines_map()
	with gfile.GFile(movie_conversations, mode="rb") as data_file:
		with gfile.GFile(output_path, mode="w") as tokens_file:
			conversation_id = 0
			num_greater_than_three = 0
			for line in data_file:
				split = line.split(split_string)
				conversation_lines = split[-1].strip().replace("'","")[1:-1].split(",") #Gets a list of keys for lines
				if( len(conversation_lines) > 2 ):
					num_greater_than_three += 1
				conversation_lines = map(lambda x: line_map[x.strip()].strip(),conversation_lines)

				conversation = split_string.join(conversation_lines)
				conversation = "c%d%s%s\n" %(conversation_id,split_string,conversation)
				tokens_file.write(conversation)
				conversation_id += 1
	print num_greater_than_three
	print conversation_id




