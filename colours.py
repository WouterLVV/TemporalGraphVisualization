import csv

# a sequence of 15 xkcd colours which go well together
colour_palette = [
	"watermelon",
	"soft green",
	"pink",
	"dark sky blue",
	"light purple",
	"turquoise",
	"brownish pink",
	"dandelion",
	"sky",
	"pale orange",
	"cool grey",
	"mango",
	"turquoise blue",
	"deep lavender",
	"lemon green"
]

def read_colours(fname):
	colour_dict = {}

	with open(fname) as csvfile:
		colour_file = csv.reader(csvfile, delimiter='\t')
		for row in colour_file: # row: dusty lavender	#ac86a8
			colour_name = row[0]
			rgb_string  = row[1]
			colour_dict[colour_name] = rgb_string
	return colour_dict

def convert_rgb_hexstring_to_tuple(rgb_string):
	r = int(rgb_string[1:3], 16) / 256
	g = int(rgb_string[3:5], 16) / 256
	b = int(rgb_string[5: ], 16) / 256

	return r, g, b

def assign_colours_rgba_tuple(items, alpha=1):
	fname = "xkcd_colours.csv"
	colour_dict = read_colours(fname)

	colours_assigned = {} # {'2BIO1': (1,0,0,1), ...}

	for i in range(len(items)):
		c_list = list(convert_rgb_hexstring_to_tuple(colour_dict[colour_palette[i]])) + [alpha]
		colours_assigned[items[i]] = tuple(c_list)

	return colours_assigned

if __name__ == '__main__':
	print(assign_colours_rgba_tuple(['2BIO1', '2BIO2', '2BIO3', 'MP', 'MPst1', 'MPst2', 'PC', 'PCst', 'PSIst']))
