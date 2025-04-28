#####
Download the needed .csv here = https://github.com/ToniGomezPi/SteamRecommendation/releases/tag/IA
Couldnt upload the .csv as it's 380MB and GitHub doesnt support more than 100MB
#####
If you want to start from scratch, Delete all csv except for merged_data
#####
You'll need MySQL once you have the latest csv generated in which you'll Create a table:
CREATE TABLE steamgamesdb (
	title TEXT,
    recommendations TEXT
);
After you can proceed to table data import wizard. at some point it'll show a unhandle list error, click on OK -> options -> field separator = "," and enclose strings " -> Next and import (took 5 minutes to me ).
