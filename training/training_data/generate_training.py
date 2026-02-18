import json

#script to create a file with 75 cities for geographic training data phrases.
# 75 example cities worldwide
cities = [
    "London", "New York", "Tokyo", "Paris", "Boston", "Sydney", "Moscow", "Toronto",
    "San Francisco", "Singapore", "Los Angeles", "Chicago", "Madrid", "Rome", "Bangkok",
    "Hong Kong", "Barcelona", "San Francisco", "Amsterdam", "Seoul", "Istanbul",
    "Mexico City", "Mumbai", "Beijing", "Buenos Aires", "Orlando", "Cairo",
    "Lagos", "Lima", "Vienna", "Stockholm", "Lisbon", "Zurich", "Prague", "Budapest",
    "Warsaw", "Dublin", "Helsinki", "Oslo", "Brussels", "Copenhagen", "Athens",
    "Kuala Lumpur", "Los Angeles", "Washington, D.C.", "Santiago", "Bogota", "Caracas", "Johannesburg",
    "Melbourne", "Montreal", "Vancouver", "Edinburgh", "Glasgow", "Birmingham", "Manchester",
    "Munich", "Frankfurt", "Hamburg", "Barcelona", "Valencia", "Seville", "Naples", "Milan",
    "Venice", "Florence", "Turin", "Napier", "Wellington", "Auckland", "Perth", "Adelaide",
    "Brisbane", "Hanoi", "Raleigh", "Riyadh", "Doha", "Kuwait City", "Muscat", "Baghdad"
]

# 7 phrase variations
phrases = [
    "I'm interested in the location of {}.",
    "Search geographic area of {}.",
    "Register location for {}.",
    "Search the destination {}.",
    "Find me information on {}.",
    "Set up new area for {}.",
    "I'd like to travel to {}."
]

# Repeat to reach ~2625 lines: 75 cities * 7 phrases * 5 repeats = 2625
repeats = 5

lines = []
for _ in range(repeats):
    for city in cities:
        for phrase in phrases:
            line = {
                "input": phrase.format(city),
                "assistant_tool_json": f"\n{{\n \"tool\":\"geographic_interest\",\n \"arguments\":{{\n \"location\":\":\"{city}\"\n }}\n}}\n"
            }
            lines.append(line)

# Save to JSONL file
with open("CaseA.jsonl", "w", encoding="utf-8") as f:
    for item in lines:
        f.write(json.dumps(item) + "\n")

print(f"JSONL file created with {len(lines)} lines: CaseA.jsonl")
