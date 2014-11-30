"""

@author Michael Bartoli

parse_rt.py: parses the rotten tomatoes database into the following folder structure:
	| rt_movies --------------------------------
	-----------------|data ---------------------
	-------------------------| 0 ---------------
	------------------------------| %sent_id.txt
	-------------------------| 1 ---------------
	------------------------------| %sent_id.txt
	-------------------------| 2 ---------------
	------------------------------| %sent_id.txt
	-------------------------| 3 ---------------
	------------------------------| %sent_id.txt
	-------------------------| 4 ---------------
	------------------------------| %sent_id.txt

Args:
	sys[1]:	path of rotten tomatoes data

"""

import sys

def main(path_to_data):
	f = open(path_to_data, "r")
	c = 0
	for l in f:
		if c > 0:
			e = l.split("\t")
			case = e[3][0]
			fname = e[0]+".txt"
			review = e[2]
			new_path = case+"/"+fname
			w = open(new_path, "w")
			w.write(review)
			w.close()
		c += 1

if __name__ == "__main__":
	path = sys.argv[1]
	main(path)

