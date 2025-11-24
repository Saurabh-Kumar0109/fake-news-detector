import argparse
import os
import sys
import pandas as pd
import random

# Add parent directory to path for imports (if needed)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Sample fake news templates
FAKE_NEWS_TEMPLATES = [
    "BREAKING: Scientists discover {thing} can cure {disease}!",
    "You won't believe what {celebrity} said about {topic}!",
    "SHOCKING: {government} secretly planning to {action}!",
    "{Celebrity} DEAD at {age} - family confirms tragic news",
    "New study proves {food} causes {disease} - doctors shocked!",
    "{Country} to ban {thing} starting next month",
    "ALERT: {disaster} heading straight for {city}!",
    "{Celebrity} caught doing {scandalous_action} on camera",
    "Government hiding truth about {conspiracy_topic}",
    "Miracle cure: {food} can make you {benefit} in just {days} days!"
]

# Sample real news templates
REAL_NEWS_TEMPLATES = [
    "Local council approves new {infrastructure} project for {city}",
    "{Company} reports quarterly earnings, stock {direction} by {percent}%",
    "Study published in {journal} examines effects of {topic}",
    "{Government} announces new policy on {policy_area}",
    "Researchers at {university} make progress in {field} research",
    "Weather forecast: {weather} expected this weekend in {region}",
    "{Sports_team} wins against {opponent} in close match",
    "New {technology} feature released by {company}",
    "{Official} speaks at conference about {topic}",
    "Community fundraiser raises ${amount} for {cause}"
]

# Vocabulary for templates
THINGS = ["coffee", "chocolate", "water", "sunlight", "exercise", "music", "sleep", "meditation"]
DISEASES = ["cancer", "diabetes", "heart disease", "depression", "arthritis", "anxiety"]
CELEBRITIES = ["Taylor Swift", "Elon Musk", "Tom Hanks", "Jennifer Lawrence", "Brad Pitt", "Oprah Winfrey"]
TOPICS = ["climate change", "artificial intelligence", "social media", "education", "healthcare"]
GOVERNMENTS = ["US Government", "European Union", "United Nations", "Federal Reserve"]
ACTIONS = ["control the internet", "raise taxes", "ban protests", "monitor citizens"]
FOODS = ["sugar", "salt", "red meat", "processed food", "dairy products"]
COUNTRIES = ["China", "Russia", "United States", "United Kingdom", "France"]
CITIES = ["New York", "Los Angeles", "London", "Tokyo", "Sydney"]
DISASTERS = ["Hurricane Category 5", "Massive earthquake", "Tsunami wave", "Volcanic eruption"]
SCANDALOUS_ACTIONS = ["illegal activities", "unethical behavior", "controversial statements"]
CONSPIRACY_TOPICS = ["aliens", "5G towers", "vaccine ingredients", "moon landing"]
BENEFITS = ["lose weight", "look younger", "boost energy", "improve memory"]
DAYS = ["7", "10", "14", "21", "30"]

# Real news vocabulary
INFRASTRUCTURE = ["park", "bridge", "library", "transit system", "school"]
COMPANIES = ["Apple", "Microsoft", "Amazon", "Google", "Tesla", "Meta"]
DIRECTIONS = ["up", "down", "unchanged"]
PERCENTS = ["2", "5", "8", "12", "15"]
JOURNALS = ["Nature", "Science", "The Lancet", "Cell", "PNAS"]
UNIVERSITIES = ["Stanford University", "MIT", "Harvard", "Oxford", "Cambridge"]
FIELDS = ["medical", "environmental", "computer science", "physics", "biology"]
WEATHER = ["Rain", "Snow", "Clear skies", "Thunderstorms", "Fog"]
REGIONS = ["the Northeast", "California", "the Midwest", "the South", "the Pacific Northwest"]
SPORTS_TEAMS = ["Lakers", "Yankees", "Patriots", "Warriors", "Cowboys"]
OPPONENTS = ["Celtics", "Red Sox", "Giants", "Rockets", "Eagles"]
TECHNOLOGIES = ["AI", "security", "privacy", "cloud", "mobile"]
OFFICIALS = ["Mayor", "Governor", "Senator", "CEO", "President"]
POLICY_AREAS = ["education", "healthcare", "environment", "transportation", "housing"]
AMOUNTS = ["10000", "25000", "50000", "100000", "250000"]
CAUSES = ["local hospital", "animal shelter", "children's charity", "disaster relief", "education fund"]


def generate_fake_news(n: int) -> list:
    news = []
    for _ in range(n):
        template = random.choice(FAKE_NEWS_TEMPLATES)
        text = template.format(
            thing=random.choice(THINGS),
            disease=random.choice(DISEASES),
            celebrity=random.choice(CELEBRITIES),
            Celebrity=random.choice(CELEBRITIES),
            topic=random.choice(TOPICS),
            government=random.choice(GOVERNMENTS),
            Government=random.choice(GOVERNMENTS),
            action=random.choice(ACTIONS),
            age=random.randint(30, 90),
            food=random.choice(FOODS),
            Country=random.choice(COUNTRIES),
            disaster=random.choice(DISASTERS),
            city=random.choice(CITIES),
            scandalous_action=random.choice(SCANDALOUS_ACTIONS),
            conspiracy_topic=random.choice(CONSPIRACY_TOPICS),
            benefit=random.choice(BENEFITS),
            days=random.choice(DAYS)
        )
        news.append(text)
    return news


def generate_real_news(n: int) -> list:
    news = []
    for _ in range(n):
        template = random.choice(REAL_NEWS_TEMPLATES)
        text = template.format(
            infrastructure=random.choice(INFRASTRUCTURE),
            city=random.choice(CITIES),
            Company=random.choice(COMPANIES),
            company=random.choice(COMPANIES),
            direction=random.choice(DIRECTIONS),
            percent=random.choice(PERCENTS),
            journal=random.choice(JOURNALS),
            topic=random.choice(TOPICS),
            Government=random.choice(GOVERNMENTS),
            policy_area=random.choice(POLICY_AREAS),
            university=random.choice(UNIVERSITIES),
            field=random.choice(FIELDS),
            weather=random.choice(WEATHER),
            region=random.choice(REGIONS),
            Sports_team=random.choice(SPORTS_TEAMS),
            opponent=random.choice(OPPONENTS),
            technology=random.choice(TECHNOLOGIES),
            Official=random.choice(OFFICIALS),
            amount=random.choice(AMOUNTS),
            cause=random.choice(CAUSES)
        )
        news.append(text)
    return news


def generate_dataset(n_samples: int, output_path: str, test_split: float = 0.2):
    """Generate a synthetic fake news dataset with balanced classes."""
    n_per_class = n_samples // 2
    
    fake = generate_fake_news(n_per_class)
    real = generate_real_news(n_per_class)
    
    df = pd.DataFrame({
        'text': fake + real,
        'label': [0] * n_per_class + [1] * n_per_class
    })
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train and test
    split_idx = int(len(df) * (1 - test_split))
    train_df = df[:split_idx]
    test_df = df[split_idx:]
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    train_path = output_path.replace('.csv', '_train.csv')
    test_path = output_path.replace('.csv', '_test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Generated {len(train_df)} training samples -> {train_path}")
    print(f"Generated {len(test_df)} test samples -> {test_path}")
    print(f"Class distribution (train): {train_df['label'].value_counts().to_dict()}")
    
    return train_path, test_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic fake news dataset')
    parser.add_argument('--n-samples', type=int, default=1000, help='Total number of samples to generate')
    parser.add_argument('--output', type=str, default='data/sample_data.csv', help='Output CSV path')
    parser.add_argument('--test-split', type=float, default=0.2, help='Fraction for test set')
    
    args = parser.parse_args()
    generate_dataset(args.n_samples, args.output, args.test_split)
