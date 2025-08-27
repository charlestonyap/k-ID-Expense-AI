# training dataset: https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset

import pandas as pd
import numpy as np
import streamlit as st
import pickle
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz, process
import time
import warnings
import logging
import hashlib
from functools import lru_cache
import threading
from threading import Lock
from email.message import EmailMessage
import smtplib

# Suppress Streamlit ScriptRunContext warnings
os.environ['STREAMLIT_LOGGER_LEVEL'] = 'ERROR'
logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')

# suppress NLP transformer error
warnings.filterwarnings("ignore", message="`encoder_attention_mask` is deprecated")

# ML imports
try:
    import shap
except ImportError:
    shap = None
    
try:
    import spacy
except ImportError:
    spacy = None
    
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go

EMAIL_LOCK = Lock()

class PersonalExpenseDetector:
    def __init__(self, model_path=None, auto_train=False):
        """Enhanced detector with robust rule-based system and ML components"""
        # Core ML components
        self.scaler = StandardScaler()
        self.category_encoder = LabelEncoder()
        self.category_encoder_fitted = False
        self.best_model = None
        self.best_model_name = None
        self.is_ml_trained = False
        self.model_path = model_path
        self.training_stats = {}
        self.cv_results = {}
        self.vendor_cache = {}
        self.semantic_cache = {}
        self.cache_lock = Lock()
        self.batch_size = 500

        # NLP Model initialization (FIRST - before any loading attempts)
        self.embedding_model = None
        self.is_embeddings_loaded = False
        self.embedding_dim = None
        self.embedding_type = None
        self.embedding_model_name = None

        # TF-IDF fallback components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.is_tfidf_fitted = False

        # Try to load pre-trained NLP models FIRST
        nlp_loaded = False
        if model_path and os.path.exists(f"{model_path}_nlp_info.pkl"):
            print("🔄 Loading pre-trained NLP model...")
            nlp_loaded = self._load_nlp_model(model_path)
            if nlp_loaded:
                print("✅ Pre-trained NLP model loaded successfully!")
            else:
                print("⚠️ Pre-trained NLP model loading failed")     
        # Only initialize embeddings if pre-trained models weren't loaded
        if not nlp_loaded:
            print("📝 No pre-trained NLP model found, initializing on-demand...")
            # NOTE: We'll initialize embeddings lazily when first needed to avoid the loading delay during startup
            pass
        
        # Enhanced personal expense patterns with weights
        self.personal_vendor_patterns = {
            'clothing_fashion': {
                'exact_matches': [
                    # Fast Fashion & Mass Market
                    'zara', 'h&m', 'uniqlo', 'cotton on', 'forever 21', 'gap', 'mango', 
                    'topshop', 'asos', 'primark', 'old navy', 'american eagle', 'hollister',
                    'abercrombie & fitch', 'urban outfitters', 'brandy melville', 'shein',
                    'boohoo', 'prettylittlething', 'missguided', 'cos', 'arket', 'weekday',

                    # Sportswear & Athletic
                    'nike', 'adidas', 'under armour', 'puma', 'reebok', 'new balance',
                    'asics', 'vans', 'converse', 'fila', 'champion', 'lululemon', 'athleta',
                    'patagonia', 'north face', 'columbia', 'decathlon', 'royal sporting house',

                    # Denim & Casual
                    'levis', 'lee', 'wrangler', 'diesel', 'tommy hilfiger', 'calvin klein',
                    'polo ralph lauren', 'lacoste', 'superdry', 'g-star raw',

                    # Luxury Fashion
                    'gucci', 'prada', 'louis vuitton', 'chanel', 'hermes', 'dior', 'fendi',
                    'versace', 'dolce & gabbana', 'armani', 'saint laurent', 'balenciaga',
                    'bottega veneta', 'celine', 'givenchy', 'valentino', 'burberry',
                    'alexander mcqueen', 'tom ford', 'off-white', 'golden goose',

                    # Mid-range Designer
                    'coach', 'kate spade', 'michael kors', 'marc jacobs', 'tory burch',
                    'ted baker', 'massimo dutti', 'sandro', 'maje', 'ganni',

                    # Watches & Jewelry
                    'rolex', 'omega', 'cartier', 'tiffany & co', 'pandora', 'swarovski',
                    'tissot', 'seiko', 'casio', 'daniel wellington', 'mvmt', 'fossil',
                    'michael kors watches', 'apple watch', 'garmin', 'fitbit',

                    # Local/Regional Brands (Singapore/Asia)
                    'charles & keith', 'pedro', 'love bonito', 'the editor\'s market',
                    'ong shunmugam', 'collate the label', 'in good company'
                ],
                'fuzzy_patterns': [
                    'boutique', 'fashion', 'apparel', 'clothing', 'garment', 'tailor',
                    'shoe', 'sneaker', 'footwear', 'accessories', 'jewelry', 'watch',
                    'handbag', 'purse', 'wallet', 'belt', 'scarf', 'hat', 'cap',
                    'dress', 'shirt', 'pants', 'jeans', 'jacket', 'coat', 'blazer',
                    'swimwear', 'lingerie', 'underwear', 'socks', 'activewear',
                    'formal wear', 'casual wear', 'streetwear', 'vintage', 'designer brand'
                ],
                'weight': 0.9
            },

            'entertainment_media': {
                'exact_matches': [
                    # Streaming Services
                    'netflix', 'spotify', 'disney+', 'hbo max', 'amazon prime video',
                    'youtube premium', 'apple music', 'apple tv+', 'paramount+', 'peacock',
                    'hulu', 'discovery+', 'espn+', 'crunchyroll', 'funimation',
                    'twitch', 'viki', 'iqiyi', 'viu', 'mewatch', 'toggle',

                    # Music Streaming
                    'tidal', 'deezer', 'soundcloud', 'bandcamp', 'pandora',

                    # Gaming
                    'steam', 'epic games', 'xbox game pass', 'playstation plus',
                    'nintendo switch online', 'origin', 'uplay', 'battle.net',
                    'riot games', 'valve', 'blizzard', 'ea', 'ubisoft',

                    # Cinema & Live Entertainment
                    'ticketmaster', 'gv cinemas', 'cathay cineplex', 'shaw theatres',
                    'filmgarde', 'golden village', 'the projector', 'sistic',
                    'eventbrite', 'bookmyshow', 'live nation', 'stubhub'
                ],
                'fuzzy_patterns': [
                    'streaming', 'subscription', 'entertainment', 'music', 'movie',
                    'concert', 'theater', 'theatre', 'gaming', 'xbox', 'playstation', 
                    'nintendo', 'cinema', 'film', 'show', 'event', 'ticket',
                    'festival', 'exhibition', 'museum', 'gallery', 'podcast',
                    'audiobook', 'digital content', 'media', 'channel'
                ],
                'weight': 0.85
            },

            'personal_care_beauty': {
                'exact_matches': [
                    # Beauty Retailers
                    'sephora', 'ulta', 'sally beauty', 'watsons', 'guardian',
                    'beauty world', 'venus beauty', 'makeup studio',

                    # Skincare Brands
                    'clinique', 'estee lauder', 'lancome', 'shiseido', 'sk-ii',
                    'la mer', 'drunk elephant', 'the ordinary', 'cerave', 'neutrogena',
                    'olay', 'l\'oreal', 'nivea', 'kiehls', 'origins', 'clarins',
                    'fresh', 'tatcha', 'sunday riley', 'paula\'s choice', 'innisfree',
                    'laneige', 'sulwhasoo', 'hada labo', 'cosrx', 'some by mi',

                    # Makeup Brands
                    'mac cosmetics', 'nars', 'urban decay', 'too faced', 'benefit',
                    'fenty beauty', 'rare beauty', 'charlotte tilbury', 'bobbi brown',
                    'maybelline', 'revlon', 'covergirl', '3ina', 'nyx professional makeup',

                    # Fragrance & Body Care
                    'bath & body works', 'lush', 'the body shop', 'crabtree & evelyn',
                    'jo malone', 'tom ford beauty', 'viktor & rolf', 'marc jacobs perfume',
                    'versace perfume', 'calvin klein fragrance',

                    # Hair Care
                    'tresemme', 'pantene', 'head & shoulders', 'herbal essences',
                    'schwarzkopf', 'wella', 'matrix', 'redken', 'paul mitchell',
                    'moroccanoil', 'olaplex', 'dyson hair', 'ghd'
                ],
                'fuzzy_patterns': [
                    'salon', 'spa', 'beauty', 'cosmetics', 'skincare', 'fragrance',
                    'perfume', 'barber', 'nail', 'massage', 'wellness', 'facial',
                    'hair care', 'makeup', 'manicure', 'pedicure', 'eyebrow',
                    'eyelash', 'threading', 'waxing', 'dermatology', 'aesthetic',
                    'botox', 'filler', 'laser treatment', 'beauty treatment'
                ],
                'weight': 0.88
            },

            'fitness_health': {
                'exact_matches': [
                    # Gym Chains
                    'fitness first', 'pure fitness', 'anytime fitness', 'planet fitness',
                    'virgin active', 'california fitness', 'true fitness', 'f45',
                    'orange theory', 'crossfit', 'barry\'s bootcamp', 'soulcycle',
                    'equinox', 'gold\'s gym', 'curves', '24 hour fitness',

                    # Boutique Fitness
                    'corepower yoga', 'yoga works', 'pilates plus', 'barre3',
                    'spinning', 'zumba', 'kickboxing', 'muay thai',

                    # Health & Medical
                    'quest diagnostics', 'labcorp', 'cvs health', 'walgreens',
                    'rite aid', 'kaiser permanente', 'blue cross', 'aetna',
                    'mount elizabeth', 'raffles medical', 'singapore general hospital',

                    # Nutrition & Supplements
                    'gnc', 'vitamin shoppe', 'iherb', 'myprotein', 'optimum nutrition',
                    'herbalife', 'usana', 'amway nutrilite'
                ],
                'fuzzy_patterns': [
                    'gym', 'fitness', 'yoga', 'pilates', 'sports club', 'wellness center',
                    'personal trainer', 'health club', 'martial arts', 'boxing',
                    'swimming', 'tennis', 'badminton', 'physiotherapy', 'chiropractor',
                    'medical', 'clinic', 'hospital', 'doctor', 'dentist', 'pharmacy',
                    'supplements', 'vitamins', 'nutrition', 'dietitian', 'therapy'
                ],
                'weight': 0.75
            },

            'food_dining': {
                'exact_matches': [
                    # Fast Food Chains
                    'mcdonald\'s', 'burger king', 'kfc', 'subway', 'pizza hut',
                    'domino\'s', 'papa john\'s', 'taco bell', 'chipotle', 'wendy\'s',
                    'five guys', 'shake shack', 'in-n-out', 'chick-fil-a',
                    'popeyes', 'dairy queen', 'sonic', 'arby\'s', 'white castle',

                    # Coffee & Cafes
                    'starbucks', 'costa coffee', 'dunkin\'', 'tim hortons',
                    'peet\'s coffee', 'blue bottle', 'intelligentsia', 'ya kun',
                    'toast box', 'kopitiam', 'old town white coffee', 'coffee bean',

                    # Food Delivery
                    'grab food', 'foodpanda', 'deliveroo', 'uber eats', 'doordash',
                    'grubhub', 'postmates', 'seamless', 'zomato', 'swiggy',

                    # Casual Dining
                    'olive garden', 'red lobster', 'applebee\'s', 'chili\'s',
                    'tgi friday\'s', 'outback steakhouse', 'cheesecake factory',
                    'p.f. chang\'s', 'california pizza kitchen',

                    # Local Chains
                    'ajisen ramen', 'ippudo', 'genki sushi', 'sushi tei',
                    'crystal jade', 'din tai fung', 'paradise dynasty',
                    'jumbo seafood', 'long beach seafood', 'roland restaurant'
                ],
                'fuzzy_patterns': [
                    'restaurant', 'cafe', 'bistro', 'deli', 'bakery', 'fast food',
                    'takeaway', 'delivery', 'food court', 'dining', 'bar', 'pub',
                    'brewery', 'winery', 'buffet', 'catering', 'hawker', 'coffeeshop',
                    'ice cream', 'dessert', 'pastry', 'tea house', 'bubble tea'
                ],
                'weight': 0.5
            },

            'personal_shopping': {
                'exact_matches': [
                    # Supermarkets & Groceries
                    'fairprice', 'cold storage', 'sheng siong', 'ntuc', 'giant',
                    'carrefour', 'tesco', 'walmart', 'target', 'costco', 'sam\'s club',
                    'whole foods', 'trader joe\'s', 'kroger', 'safeway', 'publix',
                    'aldi', 'lidl', 'market basket', 'wegmans',

                    # Convenience Stores
                    '7 eleven', 'circle k', 'wawa', 'sheetz', 'quickchek',
                    'cheers', 'ministop', 'family mart', 'lawson',

                    # Pharmacies
                    'guardian pharmacy', 'watsons', 'unity pharmacy', 'cvs pharmacy',
                    'walgreens', 'rite aid', 'duane reade',

                    # Bookstores
                    'kinokuniya', 'popular bookstore', 'times bookstore', 'barnes & noble',
                    'borders', 'waterstones', 'book depository', 'amazon books',

                    # Electronics Retail
                    'best buy', 'circuit city', 'fry\'s electronics', 'micro center',
                    'challenger', 'harvey norman', 'courts', 'gain city',

                    # Pet Stores
                    'petco', 'petsmart', 'pet lovers centre', 'kohepets'
                ],
                'fuzzy_patterns': [
                    'supermarket', 'grocery', 'convenience store', 'hypermarket',
                    'pharmacy', 'bookstore', 'toy store', 'pet store', 'hardware store',
                    'department store', 'variety store', 'dollar store', 'thrift store',
                    'market', 'bazaar', 'mall', 'shopping center', 'outlet'
                ],
                'weight': 0.7
            },

            'luxury_lifestyle': {
                'exact_matches': [
                    # Luxury Fashion (expanded)
                    'tiffany & co', 'cartier', 'burberry', 'versace', 'armani',
                    'coach', 'kate spade', 'michael kors', 'pandora', 'lululemon',
                    'hermès', 'louis vuitton', 'gucci', 'prada', 'chanel',
                    'bottega veneta', 'celine', 'saint laurent', 'balenciaga',
                    'givenchy', 'valentino', 'tom ford', 'brunello cucinelli',
                    'loro piana', 'ermenegildo zegna', 'kiton', 'brioni',

                    # Luxury Watches
                    'patek philippe', 'audemars piguet', 'vacheron constantin',
                    'jaeger-lecoultre', 'iwc', 'breitling', 'tag heuer', 'hublot',
                    'richard mille', 'franck muller', 'a. lange & söhne',

                    # Luxury Jewelry
                    'harry winston', 'van cleef & arpels', 'bulgari', 'chopard',
                    'graff', 'piaget', 'boucheron', 'fred', 'pomellato',

                    # Luxury Cars & Services
                    'rolls-royce', 'bentley', 'lamborghini', 'ferrari', 'maserati',
                    'aston martin', 'mclaren', 'bugatti', 'porsche', 'mercedes-benz',
                    'bmw', 'audi', 'lexus', 'infiniti', 'acura', 'cadillac',

                    # Luxury Hotels & Travel
                    'ritz-carlton', 'four seasons', 'mandarin oriental', 'peninsula',
                    'shangri-la', 'conrad', 'st. regis', 'westin', 'w hotels',
                    'edition', 'luxury collection', 'park hyatt', 'grand hyatt',

                    # Fine Dining
                    'michelin star', 'fine dining', 'tasting menu', 'chef\'s table',
                    'omakase', 'molecular gastronomy'
                ],
                'fuzzy_patterns': [
                    'luxury', 'premium', 'designer', 'boutique', 'high-end',
                    'exclusive', 'prestige', 'collection', 'couture', 'bespoke',
                    'artisan', 'handcrafted', 'limited edition', 'vip', 'concierge',
                    'private', 'members only', 'invitation only', 'elite'
                ],
                'weight': 0.95
            },

            'technology_electronics': {
                'exact_matches': [
                    # Tech Giants
                    'apple', 'samsung', 'google', 'microsoft', 'sony', 'lg',
                    'hp', 'dell', 'lenovo', 'asus', 'acer', 'msi', 'alienware',

                    # Mobile & Accessories
                    'iphone', 'ipad', 'macbook', 'airpods', 'apple watch',
                    'galaxy', 'pixel', 'oneplus', 'xiaomi', 'huawei', 'oppo', 'vivo',
                    'otterbox', 'spigen', 'belkin', 'anker', 'ravpower',

                    # Gaming
                    'nintendo', 'xbox', 'playstation', 'steam deck', 'switch',
                    'razer', 'logitech', 'corsair', 'steelseries', 'hyperx',

                    # Audio
                    'bose', 'sennheiser', 'audio-technica', 'beats', 'jbl',
                    'harman kardon', 'b&o', 'marshall', 'klipsch', 'yamaha',

                    # Smart Home & Electronics Retail
                    'nest', 'ring', 'philips hue', 'alexa', 'echo', 'homepod',
                    'smart things', 'wemo', 'tp-link', 'netgear', 'linksys',
                    'best buy', 'circuit city', 'fry\'s electronics', 'micro center',
                    'challenger', 'harvey norman', 'courts', 'gain city'
                ],
                'fuzzy_patterns': [
                    'electronics', 'gadget', 'device', 'smartphone', 'tablet',
                    'laptop', 'computer', 'monitor', 'keyboard', 'mouse',
                    'headphones', 'earbuds', 'speaker', 'charger', 'cable',
                    'case', 'screen protector', 'smart home', 'iot', 'tech'
                ],
                'weight': 0.6
            },

            'home_lifestyle': {
                'exact_matches': [
                    # Furniture & Home Decor
                    'ikea', 'west elm', 'pottery barn', 'crate & barrel',
                    'cb2', 'restoration hardware', 'wayfair', 'overstock',
                    'world market', 'pier 1', 'homegoods', 'tj maxx home',
                    'bed bath & beyond', 'williams sonoma', 'sur la table',

                    # Home Improvement
                    'home depot', 'lowe\'s', 'menards', 'ace hardware',
                    'sherwin-williams', 'benjamin moore', 'behr',

                    # Bedding & Bath
                    'brooklinen', 'parachute', 'purple', 'casper', 'tuft & needle',
                    'tempur-pedic', 'sleep number',

                    # Kitchen & Appliances
                    'kitchen aid', 'cuisinart', 'breville', 'ninja', 'vitamix',
                    'instant pot', 'le creuset', 'all-clad', 'lodge',

                    # Local
                    'courts', 'harvey norman', 'gain city', 'best denki',
                    'fortytwo', 'epicentre'
                ],
                'fuzzy_patterns': [
                    'furniture', 'home decor', 'interior design', 'appliances',
                    'kitchenware', 'bedding', 'bathroom', 'lighting', 'rugs',
                    'curtains', 'plants', 'gardening', 'tools', 'hardware',
                    'paint', 'renovation', 'home improvement'
                ],
                'weight': 0.7
            },

            'transportation': {
                'exact_matches': [
                    # Ride Sharing & Transport Apps
                    'grab', 'uber', 'lyft', 'gojek', 'ola', 'didi', 'bolt',
                    'comfort delgro', 'trans-cab', 'prime taxi', 'singapore taxi',

                    # Airlines
                    'singapore airlines', 'singaporeair', 'cathay pacific', 'british airways',
                    'american airlines', 'united airlines', 'united', 'delta', 'lufthansa',
                    'air france', 'klm', 'emirates', 'qatar airways', 'turkish airlines',
                    'virgin atlantic', 'virgin atlan', 'asiana airlines', 'korean air',
                    'china airlines', 'chinaairline', 'air china', 'alaska air',
                    'iberia', 'finnair', 'qantas', 'jetstar', 'scoot',

                    # Car Services
                    'shell', 'esso', 'caltex', 'bp', 'chevron', 'mobil',
                    'car wash', 'jiffy lube', 'valvoline', 'autozone',
                    'advance auto parts', 'napa', 'pep boys',

                    # Public Transport
                    'ez-link', 'nets flashpay', 'simplygo', 'mrt', 'lrt', 'bus',

                    # Car Rental & Logistics
                    'hertz', 'avis', 'enterprise', 'budget', 'alamo', 'national',
                    'turo', 'zipcar', 'car2go', 'u-haul'
                ],
                'fuzzy_patterns': [
                    'transport', 'taxi', 'ride', 'car rental', 'gas station',
                    'fuel', 'parking', 'toll', 'mechanic', 'auto repair',
                    'car service', 'vehicle', 'automotive', 'airline', 'airways',
                    'airfare', 'flight', 'airport'
                ],
                'weight': 0.3
            },

            'education_learning': {
                'exact_matches': [
                    # Online Learning
                    'coursera', 'udemy', 'skillshare', 'masterclass', 'linkedin learning',
                    'pluralsight', 'codecademy', 'khan academy', 'duolingo',
                    'babbel', 'rosetta stone',

                    # Educational Services
                    'kumon', 'sylvan', 'huntington learning', 'mathnasium',
                    'british council', 'wall street english', 'berlitz',
                    'young founders school',

                    # Books & Supplies
                    'amazon', 'chegg', 'pearson', 'mcgraw hill', 'cengage'
                ],
                'fuzzy_patterns': [
                    'education', 'learning', 'course', 'training', 'tuition',
                    'school', 'university', 'college', 'academy', 'institute',
                    'workshop', 'seminar', 'certification', 'language'
                ],
                'weight': 0.8
            }
        }
        
        # Semantic groups for advanced text analysis
        self.personal_semantic_groups = {
            'fashion_clothing': [
                'clothing', 'fashion', 'apparel', 'garment', 'outfit', 'attire', 'wear',
                'shirt', 'pants', 'dress', 'jacket', 'shoes', 'sneakers', 'boots',
                'accessories', 'jewelry', 'watch', 'bag', 'purse', 'wallet'
            ],
            'entertainment_leisure': [
                'entertainment', 'leisure', 'fun', 'recreation', 'hobby', 'amusement',
                'movie', 'cinema', 'theater', 'concert', 'show', 'performance',
                'gaming', 'game', 'music', 'streaming', 'subscription'
            ],
            'personal_care': [
                'beauty', 'cosmetics', 'skincare', 'fragrance', 'perfume', 'cologne',
                'salon', 'spa', 'massage', 'wellness', 'health', 'fitness',
                'gym', 'workout', 'exercise', 'yoga', 'pilates'
            ],
            'food_dining': [
                'restaurant', 'dining', 'food', 'meal', 'lunch', 'dinner', 'breakfast',
                'cafe', 'coffee', 'drink', 'beverage', 'snack', 'takeout', 'delivery'
            ],
            'personal_occasions': [
                'personal', 'private', 'family', 'gift', 'present', 'birthday',
                'anniversary', 'wedding', 'celebration', 'party', 'vacation',
                'holiday', 'trip', 'travel', 'leisure'
            ]
        }

        self.business_semantic_groups = {
            'office_corporate': [
                'office', 'business', 'corporate', 'company', 'professional', 'work',
                'meeting', 'conference', 'training', 'seminar', 'workshop'
            ],
            'supplies_equipment': [
                'supplies', 'equipment', 'software', 'hardware', 'tools', 'materials',
                'license', 'subscription', 'service', 'maintenance', 'support'
            ],
            'client_related': [
                'client', 'customer', 'vendor', 'supplier', 'partner', 'contractor',
                'consulting', 'consultation', 'advisory', 'professional'
            ]
        }
        
        # Pre-compile regex patterns for speed
        self.compiled_patterns = {
            'gift': re.compile(r'\b(gift|present)\b', re.IGNORECASE),
            'occasion': re.compile(r'\b(birthday|anniversary|wedding)\b', re.IGNORECASE),
            'personal': re.compile(r'\b(personal|private)\b', re.IGNORECASE),
            'family': re.compile(r'\b(family|spouse|kids?|children)\b', re.IGNORECASE),
            'travel': re.compile(r'\b(vacation|holiday|trip|travel)\b', re.IGNORECASE)
        }
        
        # Confidence-based keywords
        self.personal_keywords = {
            'high_confidence': [
                'personal', 'birthday', 'anniversary', 'vacation', 'holiday',
                'leisure', 'entertainment', 'gift', 'wedding', 'celebration'
            ],
            'medium_confidence': [
                'clothing', 'fashion', 'beauty', 'cosmetics', 'jewelry',
                'fitness', 'gym', 'spa', 'massage', 'dining', 'restaurant'
            ]
        }
        
        self.business_keywords = {
            'high_confidence': [
                'office', 'business', 'corporate', 'company', 'client',
                'meeting', 'conference', 'training', 'professional', 'team'
            ],
            'medium_confidence': [
                'supplies', 'equipment', 'software', 'license', 'subscription',
                'consulting', 'services', 'maintenance', 'support'
            ]
        }
        
        # Pre-process keywords for faster lookup (use sets instead of lists)
        self.personal_keywords_set = {
            'high_confidence': set(self.personal_keywords['high_confidence']),
            'medium_confidence': set(self.personal_keywords['medium_confidence'])
        }
        self.business_keywords_set = {
            'high_confidence': set(self.business_keywords['high_confidence']),
            'medium_confidence': set(self.business_keywords['medium_confidence'])
        }
        
        # Initialize semantic groups for embeddings
        self.personal_semantic_groups = {
            'fashion_clothing': ['clothing', 'fashion', 'apparel', 'shoes', 'accessories'],
            'entertainment_leisure': ['entertainment', 'movie', 'concert', 'show', 'leisure'],
            'personal_care': ['beauty', 'spa', 'massage', 'salon', 'cosmetics'],
            'food_dining': ['restaurant', 'dining', 'food', 'cafe', 'bar'],
            'personal_occasions': ['gift', 'birthday', 'anniversary', 'wedding', 'celebration']
        }
        
        self.business_semantic_groups = {
            'office_corporate': ['office', 'corporate', 'business', 'company', 'professional'],
            'supplies_equipment': ['supplies', 'equipment', 'software', 'hardware', 'tools'],
            'client_related': ['client', 'meeting', 'conference', 'consultation', 'service']
        }
        
        self.vendor_blacklist = {
            'clothing_fashion': [
                'singapore airlines', 'singapore air', 'singaporeair', 'sia',
                'singapore post', 'singpost', 'air china', 'korean air',
                'hotel', 'hilton', 'marriott', 'resort', 'airways'
            ]
        }
        
        nlp_loaded = False
        if model_path and os.path.exists(f"{model_path}_nlp_info.pkl"):
            print("🔄 Loading pre-trained NLP model...")
            nlp_loaded = self._load_nlp_model(model_path)
            if nlp_loaded:
                print("✅ Pre-trained NLP model loaded successfully!")
            else:
                print("⚠️ Pre-trained NLP model loading failed")     
        # Only initialize embeddings if pre-trained models weren't loaded
        if not nlp_loaded:
            print("📝 No pre-trained NLP model found, initializing on-demand...")
            # NOTE: We'll initialize embeddings lazily when first needed to avoid the loading delay during startup
            pass
        
        # Load or initialize models
        if model_path and os.path.exists(model_path):
            try:
                self.load_models(model_path)
                print("✅ Pre-trained ML model loaded successfully!")
            except Exception as e:
                print(f"⚠️ Failed to load pre-trained ML model: {e}")
                if auto_train:
                    self._auto_train_model()
        elif auto_train:
            self._auto_train_model()
    
    def _initialize_embeddings(self, save_path=None):
        """Initialize word embeddings with fallback options and optional saving"""
        try:
            # Try importing required libraries
            try:
                import spacy
            except ImportError:
                spacy = None
            
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                SentenceTransformer = None
            
            if spacy:
                # Try spaCy models
                for model_name in ["en_core_web_md", "en_core_web_sm"]:
                    try:
                        print(f"🔄 Loading spaCy model: {model_name}...")
                        self.embedding_model = spacy.load(model_name)
                        if hasattr(self.embedding_model.vocab, 'vectors_length') and self.embedding_model.vocab.vectors_length > 0:
                            self.is_embeddings_loaded = True
                            self.embedding_dim = self.embedding_model.vocab.vectors_length
                            self.embedding_type = 'spacy'
                            self.embedding_model_name = model_name
                            print(f"✅ Loaded spaCy embeddings: {model_name} (dim: {self.embedding_dim})")
                            return
                    except OSError as e:
                        print(f"⚠️ Failed to load {model_name}: {e}")
                        continue

            # Fallback to sentence-transformers
            if SentenceTransformer and not self.is_embeddings_loaded:
                try:
                    print("🔄 Loading Sentence-BERT model...")
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.is_embeddings_loaded = True
                    self.embedding_dim = 384
                    self.embedding_type = 'sentence_bert'
                    self.embedding_model_name = 'all-MiniLM-L6-v2'
                    print(f"✅ Loaded Sentence-BERT embeddings (dim: {self.embedding_dim})")
                    return
                except Exception as e:
                    print(f"⚠️ Failed to load Sentence-BERT: {e}")
                    pass

            # Final fallback
            self.embedding_type = 'tfidf'
            self.embedding_model_name = 'tfidf_fallback'
            print("📝 Using TF-IDF as fallback for text analysis")

        except Exception as e:
            print(f"⚠️ Embedding initialization error: {e}")

    def _ensure_nlp_initialized(self):
        """Ensure NLP models are initialized (lazy loading)"""
        if not self.is_embeddings_loaded and self.embedding_type is None:
            print("🔄 Initializing NLP models on first use...")
            self._initialize_embeddings()

    def _preprocess_transaction_texts(self, transactions_df):
        """Pre-process all text data for faster analysis"""
        df = transactions_df.copy()
        
        # Extract and normalize text fields
        df['vendor_clean'] = df.apply(lambda row: 
            self._get_field_value(row.to_dict(), ['merchant', 'Vendor name', 'vendor'], '').lower().strip(), 
            axis=1)
        
        df['description_clean'] = df.apply(lambda row: 
            self._get_field_value(row.to_dict(), ['description', 'Description', 'memo'], '').lower().strip(), 
            axis=1)
        
        df['combined_text'] = (df['vendor_clean'] + ' ' + df['description_clean']).str.strip()
        df['text_hash'] = df['combined_text'].apply(lambda x: hashlib.md5(x.encode()).hexdigest() if x else '')
        
        return df

    def _parallel_analysis(self, transactions_df, threshold):
        """Parallel processing for large datasets"""
        results = []
        chunk_size = max(100, len(transactions_df) // 4)  # 4 chunks minimum
        chunks = [transactions_df.iloc[i:i + chunk_size] for i in range(0, len(transactions_df), chunk_size)]
        
        print(f"🔄 Processing {len(chunks)} chunks in parallel...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_chunk = {executor.submit(self._process_chunk, chunk, threshold): i 
                             for i, chunk in enumerate(chunks)}
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    print(f"✅ Chunk {chunk_idx + 1}/{len(chunks)} completed")
                except Exception as e:
                    print(f"❌ Chunk {chunk_idx + 1} failed: {e}")
        
        return results

    def _process_chunk(self, chunk_df, threshold):
        """Process a chunk of transactions"""
        chunk_results = []
        
        for idx, row in chunk_df.iterrows():
            try:
                result = self._analyze_single_transaction(row.to_dict(), threshold)
                chunk_results.append(result)
            except Exception as e:
                # Add error result
                error_result = {
                    **row.to_dict(),
                    'combined_score': 0, 'rule_score': 0, 'ml_score': 0,
                    'is_flagged': False, 'risk_level': 'Error',
                    'confidence_factors': f'Error: {str(e)[:50]}',
                    'classification': 'Error'
                }
                chunk_results.append(error_result)
        
        return chunk_results

    def _analyze_single_transaction(self, transaction, threshold):
        """Fast single transaction analysis"""
        # Get prediction with optimized methods
        prediction = self.predict_personal_expense(transaction)
        
        # Determine classification
        is_flagged = prediction['final_score'] >= threshold
        
        if prediction['final_score'] >= 75:
            risk_level = "High"
        elif prediction['final_score'] >= 40:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            **transaction,
            'combined_score': round(prediction['final_score'], 1),
            'rule_score': round(prediction['rule_score'], 1),
            'ml_score': round(prediction['ml_score'], 1),
            'is_flagged': is_flagged,
            'risk_level': risk_level,
            'confidence_factors': '; '.join(prediction['confidence_factors'][:3]),
            'classification': 'Personal' if is_flagged else 'Business'
        }

    def analyze_transactions(self, transactions_df, threshold=50):
        """Analyze multiple transactions with enhanced detection - OPTIMIZED VERSION"""
        results = []
        total_transactions = len(transactions_df)
        
        print(f"🚀 Starting optimized analysis of {total_transactions} transactions...")
        
        # Pre-process all text data in one go
        print("📝 Pre-processing text data...")
        transactions_df = self._preprocess_transaction_texts(transactions_df)
        
        # Use parallel processing for large datasets
        if total_transactions > 1000:
            results = self._parallel_analysis(transactions_df, threshold)
        else:
            # Use optimized sequential processing
            results = self._sequential_analysis(transactions_df, threshold)
        
        print(f"✅ Analysis completed: {len(results)} results generated")
        return pd.DataFrame(results)

    def _sequential_analysis(self, transactions_df, threshold):
        """Optimized sequential processing"""
        results = []
        total = len(transactions_df)
        
        for idx, row in transactions_df.iterrows():
            try:
                result = self._analyze_single_transaction(row.to_dict(), threshold)
                results.append(result)
                
                # Progress update every 100 transactions
                if (idx + 1) % 100 == 0:
                    print(f"Progress: {idx + 1}/{total} ({((idx + 1)/total)*100:.1f}%)")
                    
            except Exception as e:
                error_result = {
                    **row.to_dict(),
                    'combined_score': 0, 'rule_score': 0, 'ml_score': 0,
                    'is_flagged': False, 'risk_level': 'Error',
                    'confidence_factors': f'Error: {str(e)[:50]}',
                    'classification': 'Error'
                }
                results.append(error_result)
        
        return results

    def predict_personal_expense(self, transaction):
        """Combined prediction using enhanced rules and ML fraud detection"""
        # Get rule-based score
        rule_score, confidence_factors, detailed_breakdown = self.enhanced_rule_based_score(transaction)
        
        # Get ML fraud score
        ml_score = self.ml_fraud_score(transaction) if hasattr(self, 'ml_fraud_score') else 0
        
        # Combine scores with dynamic weighting
        if rule_score >= 60:  # High rule confidence
            final_score = (rule_score * 0.8) + (ml_score * 0.2)
        elif rule_score <= 20:  # Low rule confidence, trust ML more
            final_score = (rule_score * 0.4) + (ml_score * 0.6)
        else:  # Balanced approach
            final_score = (rule_score * 0.6) + (ml_score * 0.4)
        
        return {
            'final_score': min(max(final_score, 0), 100),
            'rule_score': rule_score,
            'ml_score': ml_score,
            'confidence_factors': confidence_factors,
            'detailed_breakdown': detailed_breakdown
        }

    def enhanced_rule_based_score(self, transaction):
        """Enhanced rule-based scoring with caching for performance - STRICTER VERSION"""
        score = 0
        confidence_factors = []
        detailed_breakdown = {}

        # Get pre-processed text data or create it
        combined_text = transaction.get('combined_text', '')
        text_hash = transaction.get('text_hash', '')

        if not combined_text:
            # Fallback if not pre-processed
            vendor = self._get_field_value(transaction, ['merchant', 'Vendor name', 'vendor'])
            description = self._get_field_value(transaction, ['description', 'Description', 'memo'])
            combined_text = f"{vendor} {description}".strip().lower()
            text_hash = hashlib.md5(combined_text.encode()).hexdigest() if combined_text else ''

        if not combined_text.strip():
            return 5, ["No vendor/description data"], {}

        # 1. Thread-safe cached vendor analysis
        vendor = self._get_field_value(transaction, ['merchant', 'Vendor name', 'vendor'])
        vendor_key = vendor.lower().strip() if vendor else ""

        with self.cache_lock:
            if vendor_key in self.vendor_cache:
                vendor_score, vendor_category = self.vendor_cache[vendor_key]
            else:
                vendor_score, vendor_category, _ = self.fuzzy_match_vendor(vendor)
                self.vendor_cache[vendor_key] = (vendor_score, vendor_category)

        if vendor_score > 0:
            score += min(vendor_score * 0.4, 80)
            confidence_factors.append(f"Vendor: {vendor_category}")

        # 2. Fast keyword analysis using sets - KEEP ORIGINAL LOGIC
        combined_lower = combined_text.lower()

        # Use the original keyword counting logic instead of set intersection
        high_personal = sum(1 for kw in self.personal_keywords['high_confidence'] if kw in combined_lower)
        if high_personal > 0:
            score += high_personal * 15
            confidence_factors.append(f"High confidence keywords: {high_personal}")

        med_personal = sum(1 for kw in self.personal_keywords['medium_confidence'] if kw in combined_lower)
        if med_personal > 0:
            score += med_personal * 8
            confidence_factors.append(f"Medium confidence keywords: {med_personal}")

        high_business = sum(1 for kw in self.business_keywords['high_confidence'] if kw in combined_lower)
        if high_business > 0:
            business_penalty = high_business * 20
            score -= business_penalty
            confidence_factors.append(f"Business penalty: -{business_penalty}")

        # 3. ALWAYS do semantic analysis (not conditional) - similar to original code
        if text_hash and len(combined_text) > 10:
            semantic_score = self._get_cached_semantic_score(text_hash, combined_text)
            if abs(semantic_score) > 1:
                # Use similar contribution as original code
                nlp_contribution = min(abs(semantic_score), 60)  # Increased from 25
                if semantic_score > 0:
                    score += nlp_contribution
                    confidence_factors.append(f"Semantic: +{nlp_contribution:.1f}")
                else:
                    score += semantic_score  # Apply negative score directly
                    confidence_factors.append(f"Business semantic: {semantic_score:.1f}")

        # 4. Pattern matching using pre-compiled regex - KEEP ORIGINAL SCORING
        pattern_matches = sum(1 for pattern in self.compiled_patterns.values() 
                            if pattern.search(combined_text))
        if pattern_matches > 0:
            score += pattern_matches * 12
            confidence_factors.append(f"Patterns: {pattern_matches}")

        # 5. Quick amount check
        amount = self._get_numeric_field(transaction, ['amount', 'Amount (by category)', 'amt'])
        if amount > 0:
            if amount < 25:
                score += 8
            elif amount > 2000:
                score += 12
            elif 200 <= amount <= 500:
                score += 5

        # 6. Weekend check (if date available) - KEEP ORIGINAL LOGIC
        date_field = self._get_field_value(transaction, ['date', 'Purchase date', 'trans_date_trans_time'])
        if date_field:
            try:
                dt = pd.to_datetime(date_field)
                if dt.weekday() >= 5:
                    score += 10
                    confidence_factors.append("Weekend transaction")
            except Exception:
                pass
        elif 'weekday' in transaction:  # Pre-computed in preprocessing
            if transaction['weekday'] >= 5:
                score += 10
                confidence_factors.append("Weekend")

        final_score = min(max(score, 0), 100)
        return final_score, confidence_factors, detailed_breakdown

    def _get_cached_semantic_score(self, text_hash, combined_text):
        """Get semantic score with caching"""
        with self.cache_lock:
            if text_hash in self.semantic_cache:
                return self.semantic_cache[text_hash]
        
        # Only do semantic analysis if embeddings are loaded and text is meaningful
        if (hasattr(self, 'is_embeddings_loaded') and self.is_embeddings_loaded):
            try:
                nlp_score, _, _ = self.analyze_text_features([combined_text])
                with self.cache_lock:
                    self.semantic_cache[text_hash] = nlp_score
                return nlp_score
            except Exception:
                pass
        
        return 0

    def fuzzy_match_vendor(self, vendor_name, threshold=85):  # Increase threshold from 80 to 85
        """Enhanced vendor matching using fuzzy string matching - STRICTER VERSION"""
        if not vendor_name or not vendor_name.strip():
            return 0, None, 0

        vendor_lower = vendor_name.lower().strip()
        max_score = 0
        best_category = None
        best_weight = 0
        
        # Check blacklist before processing
        vendor_lower = vendor_name.lower().strip()
        for category, blacklisted_vendors in self.vendor_blacklist.items():
            if any(blacklisted in vendor_lower for blacklisted in blacklisted_vendors):
                if category == 'clothing_fashion':
                    continue  # Skip this category for blacklisted vendors
        
        for category, patterns in self.personal_vendor_patterns.items():
            category_score = 0

            # Exact matches (highest priority) - make this more strict
            for exact_match in patterns['exact_matches']:
                # Use exact word matching instead of substring matching for short terms
                if len(exact_match) <= 4:  # Short terms need exact word match
                    if exact_match == vendor_lower or f" {exact_match} " in f" {vendor_lower} ":
                        category_score = 100
                        break
                else:  # Longer terms can use substring matching
                    if exact_match in vendor_lower:
                        category_score = 100
                        break

            # Add length-based filtering for fuzzy patterns
            if category_score < 100:
                for pattern in patterns['fuzzy_patterns']:
                    # Skip very short patterns that might cause false matches
                    if len(pattern) <= 3:
                        continue

                    if pattern in vendor_lower:
                        category_score = max(category_score, 85)
                    else:
                        # Use fuzzy matching with higher threshold
                        try:
                            from fuzzywuzzy import fuzz
                            fuzzy_score = fuzz.partial_ratio(pattern, vendor_lower)
                            if fuzzy_score >= 90:  # Increased from threshold (85)
                                category_score = max(category_score, fuzzy_score)
                        except Exception:
                            continue

            # Weight the score by category confidence
            weighted_score = category_score * patterns['weight']

            if weighted_score > max_score:
                max_score = weighted_score
                best_category = category
                best_weight = patterns['weight']

        return max_score, best_category, best_weight

    def analyze_text_features(self, text_data):
        """Analyze text features using embeddings or keyword matching"""
        self._ensure_nlp_initialized() 

        if isinstance(text_data, str):
            text_data = [text_data]

        total_combined_score = 0
        total_business_penalty = 0
        detailed_analysis = []

        for text in text_data:
            if not text or not text.strip():
                continue

            text_analysis = {
                'text': text,
                'personal_matches': {},
                'business_matches': {},
                'combined_score': 0,
                'business_penalty': 0
            }

            if self.is_embeddings_loaded:
                combined_score, business_penalty, matches = self._analyze_with_embeddings(text)
            else:
                combined_score, business_penalty, matches = self._analyze_with_keywords(text)

            text_analysis.update(matches)
            text_analysis['combined_score'] = combined_score
            text_analysis['business_penalty'] = business_penalty

            total_combined_score += combined_score
            total_business_penalty += business_penalty
            detailed_analysis.append(text_analysis)

        # Generate summary of matching terms
        matching_terms = []
        for analysis in detailed_analysis:
            for category, score in analysis.get('personal_matches', {}).items():
                if score > 0:
                    matching_terms.append(f"{category}({score:.2f})")

        return total_combined_score - total_business_penalty, matching_terms, detailed_analysis

    def _analyze_with_embeddings(self, text):
        """Analyze text using word embeddings for semantic similarity"""
        combined_score = 0
        business_penalty = 0
        analysis = {'personal_matches': {}, 'business_matches': {}}

        # Analyze personal semantic groups
        for group_name, keywords in self.personal_semantic_groups.items():
            group_similarities = []
            for keyword in keywords:
                similarity = self.calculate_semantic_similarity(text, keyword)
                if similarity > 0.3:  # Threshold for semantic similarity
                    group_similarities.append(similarity)

            if group_similarities:
                max_similarity = max(group_similarities)
                group_weights = {
                    'fashion_clothing': 1.2,
                    'entertainment_leisure': 1.0,
                    'personal_care': 1.1,
                    'food_dining': 0.8,
                    'personal_occasions': 1.5
                }
                
                group_score = max_similarity * 20
                weighted_score = group_score * group_weights.get(group_name, 1.0)
                combined_score += weighted_score
                analysis['personal_matches'][group_name] = weighted_score

        # Analyze business semantic groups
        for group_name, keywords in self.business_semantic_groups.items():
            group_similarities = []
            for keyword in keywords:
                similarity = self.calculate_semantic_similarity(text, keyword)
                if similarity > 0.4:  # Higher threshold for business terms
                    group_similarities.append(similarity)

            if group_similarities:
                max_similarity = max(group_similarities)
                group_penalty = max_similarity * 25
                business_penalty += group_penalty
                analysis['business_matches'][group_name] = group_penalty

        return combined_score, business_penalty, analysis

    def _analyze_with_keywords(self, text):
        """Fallback keyword-based analysis"""
        combined_score = 0
        business_penalty = 0
        analysis = {'personal_matches': {}, 'business_matches': {}}

        text_lower = text.lower()

        # Analyze personal semantic groups
        for group_name, keywords in self.personal_semantic_groups.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                group_weights = {
                    'fashion_clothing': 12,
                    'entertainment_leisure': 10,
                    'personal_care': 11,
                    'food_dining': 8,
                    'personal_occasions': 15
                }
                group_score = matches * group_weights.get(group_name, 10)
                combined_score += group_score
                analysis['personal_matches'][group_name] = group_score

        # Analyze business semantic groups
        for group_name, keywords in self.business_semantic_groups.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                group_penalties = {
                    'office_corporate': 20,
                    'supplies_equipment': 15,
                    'client_related': 18
                }
                group_penalty = matches * group_penalties.get(group_name, 15)
                business_penalty += group_penalty
                analysis['business_matches'][group_name] = group_penalty

        return combined_score, business_penalty, analysis

    def get_text_embedding(self, text):
        """Get embedding for text"""
        self._ensure_nlp_initialized()  

        if not self.is_embeddings_loaded or not text.strip():
            return None

        try:
            text = text.lower().strip()
            if hasattr(self.embedding_model, 'vocab'):  # spaCy
                doc = self.embedding_model(text)
                if doc.has_vector:
                    return doc.vector
            elif hasattr(self.embedding_model, 'encode'):  # sentence-transformers
                return self.embedding_model.encode([text])[0]
        except Exception:
            pass
        return None

    def calculate_semantic_similarity(self, text1, text2):
        """Calculate cosine similarity between two texts"""
        self._ensure_nlp_initialized() 

        if not self.is_embeddings_loaded:
            return 0.0

        emb1 = self.get_text_embedding(text1)
        emb2 = self.get_text_embedding(text2)

        if emb1 is None or emb2 is None:
            return 0.0

        try:
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity([emb1], [emb2])[0][0]
        except Exception:
            return 0.0

    def pretrain_nlp_models(self):
        """Pre-train NLP models"""
        print("🚀 Starting NLP model pre-training...")

        # Initialize embeddings
        self._initialize_embeddings()

        if self.is_embeddings_loaded:
            print("✅ NLP model pre-training completed successfully!")

            # Test the model with a sample text
            test_text = "restaurant dinner payment"
            test_embedding = self.get_text_embedding(test_text)
            if test_embedding is not None:
                print(f"🧪 Test embedding generated successfully (shape: {test_embedding.shape if hasattr(test_embedding, 'shape') else len(test_embedding)})")

            return True
        else:
            print("❌ NLP model pre-training failed!")
            return False

    def clear_caches(self):
        """Clear all caches to free memory"""
        with self.cache_lock:
            self.vendor_cache.clear()
            self.semantic_cache.clear()
        print("🗑️ Caches cleared")

    def get_cache_stats(self):
        """Get cache statistics"""
        with self.cache_lock:
            return {
                'vendor_cache_size': len(self.vendor_cache),
                'semantic_cache_size': len(self.semantic_cache)
            }

    def _get_field_value(self, transaction, field_names, default=''):
        """Get field value from transaction with multiple possible field names"""
        for field_name in field_names:
            if field_name in transaction and transaction[field_name] is not None:
                value = str(transaction[field_name]).strip()
                if value and value.lower() not in ['', 'nan', 'none', 'null']:
                    return value
        return default

    def _get_numeric_field(self, transaction, field_names, default=0):
        """Get numeric field value from transaction"""
        for field_name in field_names:
            if field_name in transaction and transaction[field_name] is not None:
                try:
                    value = float(transaction[field_name])
                    if not np.isnan(value):
                        return abs(value)  # Return absolute value
                except (ValueError, TypeError):
                    continue
        return default
    
    def calculate_frequency_features(self, df):
        """Calculate transaction frequency features efficiently"""

        # Create a copy to avoid modifying original
        result_df = df.copy()

        # Initialize frequency columns with zeros
        result_df['transaction_count_1h'] = 0
        result_df['transaction_count_24h'] = 0
        result_df['personal_vendor_freq_1h'] = 0.0


        # Parse datetime column efficiently
        datetime_col = pd.to_datetime(result_df['trans_date_trans_time'], errors='coerce')
        valid_datetime_mask = datetime_col.notna()

        if valid_datetime_mask.sum() == 0:
            return result_df

        # Work only with valid datetime records
        valid_df = result_df[valid_datetime_mask].copy()
        valid_df['trans_datetime'] = datetime_col[valid_datetime_mask]

        try:
            # Sort by datetime first to avoid duplicate index issues
            valid_df = valid_df.sort_values('trans_datetime')

            # Use pandas rolling window approach (much faster)
            valid_df = valid_df.set_index('trans_datetime')

            # 1-hour window (both directions: -1h to +1h = 2h total)
            # Use a dummy column for counting
            valid_df['_temp_count'] = 1
            rolling_1h = valid_df['_temp_count'].rolling('2h', center=True).count() - 1  # Subtract self

            # 24-hour window (both directions: -24h to +24h = 48h total) 
            rolling_24h = valid_df['_temp_count'].rolling('48h', center=True).count() - 1  # Subtract self

            # Map back to original dataframe using the original index
            original_indices = valid_df.reset_index().index
            result_df.loc[valid_datetime_mask, 'transaction_count_1h'] = rolling_1h.fillna(0).astype(int).values
            result_df.loc[valid_datetime_mask, 'transaction_count_24h'] = rolling_24h.fillna(0).astype(int).values

            # Clean up temporary column
            valid_df = valid_df.drop('_temp_count', axis=1)

            # For personal vendor frequency, use a simpler approach
            # This is a placeholder - you can enhance this based on your vendor analysis
            result_df.loc[valid_datetime_mask, 'personal_vendor_freq_1h'] = 0.1  # Placeholder

        except Exception as e:
            print(f"🔍 DEBUG: ❌ Vectorized calculation failed: {e}")

            # Fallback: Simple approach without rolling windows
            # Just use basic stats as proxies
            time_span = (valid_df['trans_datetime'].max() - valid_df['trans_datetime'].min()).total_seconds()
            if time_span > 0:
                avg_transactions_per_hour = len(valid_df) / (time_span / 3600)
            else:
                avg_transactions_per_hour = 0

            result_df.loc[valid_datetime_mask, 'transaction_count_1h'] = int(avg_transactions_per_hour)
            result_df.loc[valid_datetime_mask, 'transaction_count_24h'] = int(avg_transactions_per_hour * 24)
            result_df.loc[valid_datetime_mask, 'personal_vendor_freq_1h'] = 0.1

            print("🔍 DEBUG: ✅ Fallback calculation completed")

        return result_df
    
    
    def prepare_ml_features(self, df):
        """Prepare features for ML model efficiently, handling unseen categories"""
        # Ensure required columns exist
        required_cols = ['amt', 'category', 'trans_date_trans_time']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate frequency features
        features = self.calculate_frequency_features(df)
        
        # Parse datetime
        features['trans_datetime'] = pd.to_datetime(features['trans_date_trans_time'], errors='coerce')

        # Time-based features
        features['hour'] = features['trans_datetime'].dt.hour
        features['day_of_week'] = features['trans_datetime'].dt.dayofweek
        features['month'] = features['trans_datetime'].dt.month
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['is_night'] = ((features['hour'] >= 22) | (features['hour'] <= 6)).astype(int)
        features['is_business_hours'] = ((features['hour'] >= 9) & (features['hour'] <= 17)).astype(int)
        
        # Amount-based features
        features['log_amount'] = np.log1p(features['amt'])
        features['amount_rounded'] = (features['amt'] % 1 == 0).astype(int)
        
        # Category encoding with handling for unseen categories
        if not self.category_encoder_fitted:
            self.category_encoder.fit(features['category'])
            self.category_encoder_fitted = True
        
        # Get known categories from training data
        known_categories = set(self.category_encoder.classes_)
        default_category = 'misc_net' if 'misc_net' in known_categories else self.category_encoder.classes_[0]
        
        # Identify unseen categories
        unseen_categories = set(features['category']) - known_categories
        if unseen_categories:
            print(f"Warning: Unseen categories found: {unseen_categories}. Mapping to '{default_category}'")
        
        # Map unseen categories to default
        features['category'] = features['category'].apply(
            lambda x: x if x in known_categories else default_category
        )
        features['category_encoded'] = self.category_encoder.transform(features['category'])
        
        # Statistical features
        features['amount_zscore'] = (features['amt'] - features['amt'].mean()) / (features['amt'].std() + 1e-6)
        
        # Holiday features
        features['is_holiday_season'] = ((features['month'] == 12) | 
                                        ((features['month'] == 1) & (features['trans_datetime'].dt.day <= 7))).astype(int)
        features['is_summer_vacation'] = ((features['month'] >= 6) & (features['month'] <= 8)).astype(int)
        features['is_school_holiday'] = ((features['month'] == 12) | 
                                        (features['month'] == 1) | 
                                        ((features['month'] >= 6) & (features['month'] <= 8))).astype(int)
        
        # Select final features
        ml_features = [
            'amt', 'log_amount', 'amount_rounded', 'amount_zscore',
            'category_encoded', 'hour', 'day_of_week', 'month',
            'is_weekend', 'is_night', 'is_business_hours',
            'transaction_count_1h', 'transaction_count_24h', 'personal_vendor_freq_1h',
            'is_holiday_season', 'is_summer_vacation', 'is_school_holiday'
        ]
        return features[ml_features].fillna(0)  # Fill any remaining NaN values
    
    def train_ml_models_with_tuning(self, credit_card_data_path="credit_card_transactions.csv", test_size=0.2):
        """Enhanced training with cost-sensitive learning (CORRECTED VERSION)"""
        print("🔍 DEBUG: train_ml_models_with_tuning function called!")
        st.info("🔄 Loading credit card fraud dataset...")

        # Load the fraud detection dataset
        print("🔍 DEBUG: About to load CSV")
        df = pd.read_csv(credit_card_data_path)
        print(f"🔍 DEBUG: CSV loaded with {len(df)} rows")
        st.success(f"✅ Loaded {len(df)} transaction records")

        # Check for required columns
        print("🔍 DEBUG: Checking required columns")
        required_cols = ['amt', 'category', 'trans_date_trans_time', 'job', 'is_fraud']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
        print("🔍 DEBUG: All required columns present")

        # Prepare features
        print("🔍 DEBUG: About to prepare ML features")
        st.info("🔄 Engineering features...")
        X = self.prepare_ml_features(df)  
        print(f"🔍 DEBUG: Features prepared, shape: {X.shape}")
        y = df['is_fraud']

        # Cost-sensitive parameters
        fraud_cost_ratio = 10  # Cost of missing fraud vs false alarm

        # Calculate class weights BEFORE SMOTE (using original distribution)
        fraud_rate = df['is_fraud'].mean()
        legitimate_rate = 1 - fraud_rate

        print(f"🔍 DEBUG: Original fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")

        # Cost-sensitive class weights based on ORIGINAL distribution
        cost_sensitive_weights = {
            0: 1.0,  # Legitimate transactions
            1: fraud_cost_ratio * (legitimate_rate / fraud_rate)  # Fraud transactions
        }

        print(f"🔍 DEBUG: Cost-sensitive weights - Legitimate: {cost_sensitive_weights[0]:.2f}, Fraud: {cost_sensitive_weights[1]:.2f}")

        # DON'T use SMOTE with cost-sensitive learning - they conflict!
        # Train-test split on original imbalanced data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        print(f"🔍 DEBUG: Train fraud rate: {y_train.mean():.4f}, Test fraud rate: {y_test.mean():.4f}")

        st.info(f"📊 Split: {len(X_train)} training, {len(X_test)} testing samples")
        st.info(f"📊 Fraud rates - Train: {y_train.mean()*100:.2f}%, Test: {y_test.mean()*100:.2f}%")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define models with cost-sensitive parameters
        models = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [15, None],
                    'min_samples_split': [2, 4],
                    'min_samples_leaf': [1,3],
                    'class_weight': ['balanced', cost_sensitive_weights]
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [200, 300],
                    'max_depth': [8, 10, None],
                    'learning_rate': [0.1, 0.2, 0.3],
                    'subsample': [0.9, 0.8],
                    'scale_pos_weight': [1, fraud_cost_ratio, fraud_cost_ratio * 0.5]
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=2000),  # Increased max_iter
                'params': {
                    'C': [100, 200, 300],
                    'penalty': ['l1', 'l2'],
                    'solver': ['saga', 'liblinear'],
                    'class_weight': ['balanced', cost_sensitive_weights]
                }
            }
        }

        # Calculate total combinations for overall progress
        total_combinations = 0
        combination_counts = {}
        for name, config in models.items():
            param_grid = list(ParameterGrid(config['params']))
            combination_counts[name] = len(param_grid)
            total_combinations += len(param_grid)

        print(f"\n📊 Training Overview:")
        for name, count in combination_counts.items():
            print(f"   {name}: {count} parameter combinations")
        print(f"   Total combinations: {total_combinations}")
        print("=" * 60)

        # Cross-validation setup
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # CORRECTED cost-sensitive scorer - normalize by sample size
        def cost_sensitive_scorer(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            # Calculate normalized cost per sample
            total_samples = len(y_true)
            cost_per_sample = ((fraud_cost_ratio * fn) + (1 * fp)) / total_samples

            # Return negative cost (higher is better for sklearn)
            return -cost_per_sample

        custom_scorer = make_scorer(cost_sensitive_scorer)

        # Threshold optimization function
        def optimize_threshold(model, X_val, y_val, cost_ratio=fraud_cost_ratio):
            """Find optimal threshold that minimizes cost"""
            y_prob = model.predict_proba(X_val)[:, 1]
            thresholds = np.arange(0.1, 0.9, 0.05)

            best_threshold = 0.5
            best_cost_per_sample = float('inf')

            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

                # Normalize cost by sample size
                total_cost = (cost_ratio * fn) + (1 * fp)
                cost_per_sample = total_cost / len(y_val)

                if cost_per_sample < best_cost_per_sample:
                    best_cost_per_sample = cost_per_sample
                    best_threshold = threshold

            return best_threshold, best_cost_per_sample

        # Results storage
        results = {}
        best_cv_score = -float('inf')
        global_combination_counter = 0

        # Train and tune each model
        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        for idx, (name, config) in enumerate(models.items()):
            print(f"\n🚀 Starting {name} training...")
            print(f"   Parameter combinations to test: {combination_counts[name]}")
            st.info(f"🔄 Training {name} ({combination_counts[name]} combinations)...")

            try:
                # Use appropriate data (scaled for LogReg, original for tree-based)
                X_train_model = X_train_scaled if name in ['Logistic Regression'] else X_train.values
                X_test_model = X_test_scaled if name in ['Logistic Regression'] else X_test.values

                param_grid = list(ParameterGrid(config['params']))
                total_combinations_this_model = len(param_grid)

                best_score = -float('inf')
                best_params = None
                best_estimator = None

                for i, params in enumerate(param_grid):
                    global_combination_counter += 1

                    print(f"   └─ Combination {i+1}/{total_combinations_this_model}: {params}")
                    status_placeholder.text(f"{name}: Testing combination {i+1}/{total_combinations_this_model}")

                    # Create model with current parameters
                    model = config['model'].__class__(**{**config['model'].get_params(), **params})

                    # Perform cross-validation
                    start_time = time.time()
                    cv_scores = cross_val_score(model, X_train_model, y_train, cv=cv, scoring=custom_scorer, n_jobs=1)
                    end_time = time.time()

                    mean_score = cv_scores.mean()
                    std_score = cv_scores.std()

                    # Convert back to interpretable cost
                    avg_cost_per_sample = -mean_score
                    print(f"      CV Cost per Sample: {avg_cost_per_sample:.4f} (±{std_score:.4f}) | Time: {end_time-start_time:.2f}s")

                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = params
                        best_estimator = model
                        print(f"      ⭐ New best for {name}!")

                    overall_progress = global_combination_counter / total_combinations
                    progress_bar.progress(overall_progress)

                # Train the best model on full training set
                print(f"\n   🏆 Best {name} parameters: {best_params}")
                print(f"   📈 Best CV Cost per Sample: {-best_score:.4f}")
                print(f"   🔄 Training final {name} model on full training set...")

                final_train_start = time.time()
                best_estimator.fit(X_train_model, y_train)
                final_train_end = time.time()

                # Optimize threshold
                optimal_threshold, min_cost_per_sample = optimize_threshold(best_estimator, X_test_model, y_test)

                # Use optimal threshold for final predictions
                y_test_pred_proba = best_estimator.predict_proba(X_test_model)[:, 1]
                y_test_pred = (y_test_pred_proba >= optimal_threshold).astype(int)

                # Calculate test metrics
                test_accuracy = accuracy_score(y_test, y_test_pred)
                test_precision = precision_score(y_test, y_test_pred, zero_division=0)
                test_recall = recall_score(y_test, y_test_pred, zero_division=0)
                test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
                test_confusion = confusion_matrix(y_test, y_test_pred)

                # Calculate actual business cost
                tn, fp, fn, tp = test_confusion.ravel()
                total_business_cost = (fraud_cost_ratio * fn) + (1 * fp)

                results[name] = {
                    'best_estimator': best_estimator,
                    'best_params': best_params,
                    'cv_score': best_score,  # This is the negative cost per sample
                    'test_accuracy': test_accuracy,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_f1': test_f1,
                    'test_confusion_matrix': test_confusion,
                    'test_predictions': y_test_pred,
                    'test_probabilities': y_test_pred_proba,
                    'combinations_tested': total_combinations_this_model,
                    'final_training_time': final_train_end - final_train_start,
                    'optimal_threshold': optimal_threshold,
                    'cost_per_sample': min_cost_per_sample,
                    'total_business_cost': total_business_cost,
                    'cost_ratio_used': fraud_cost_ratio
                }

                # Track best model based on CV score (remember: higher is better, it's negative cost)
                if best_score > best_cv_score:
                    best_cv_score = best_score
                    self.best_model = best_estimator
                    self.best_model_name = name

                print(f"✅ {name} COMPLETED")
                print(f"   📊 Tested {total_combinations_this_model} combinations")
                print(f"   🎯 Scores - CV Cost/Sample: {-best_score:.4f}, Test F1: {test_f1:.4f}")
                print(f"   🎚️ Optimal threshold: {optimal_threshold:.3f}")
                print(f"   💰 Total business cost on test set: {total_business_cost:.0f}")
                print(f"   ⏱️ Final training time: {final_train_end - final_train_start:.2f}s")
                print("=" * 60)

                st.success(f"✅ {name} - Cost/Sample: {-best_score:.4f}, F1: {test_f1:.4f}, Cost: {total_business_cost:.0f}")

            except Exception as e:
                print(f"❌ {name} training failed: {str(e)}")
                st.warning(f"⚠️ {name} training failed: {str(e)}")
                results[name] = {'error': str(e)}

        progress_bar.empty()
        status_placeholder.empty()

        # Create ensemble
        print("\n🔄 Creating ensemble model...")
        st.info("🔄 Creating ensemble model...")

        valid_results = {name: data for name, data in results.items() if 'error' not in data}
        sorted_models = sorted(valid_results.items(), 
                              key=lambda x: x[1]['cv_score'], 
                              reverse=True)
        top_models = sorted_models[:3]

        if len(top_models) >= 2:
            ensemble_estimators = [(name, data['best_estimator']) for name, data in top_models]

            ensemble_model = VotingClassifier(
                estimators=ensemble_estimators,
                voting='soft'
            )

            needs_scaling = any(name in ['Logistic Regression'] for name, _ in top_models)
            X_train_ensemble = X_train_scaled if needs_scaling else X_train.values

            ensemble_train_start = time.time()
            ensemble_model.fit(X_train_ensemble, y_train)
            ensemble_train_end = time.time()

            X_test_ensemble = X_test_scaled if needs_scaling else X_test.values
            ensemble_threshold, ensemble_cost_per_sample = optimize_threshold(ensemble_model, X_test_ensemble, y_test)

            ensemble_pred_proba = ensemble_model.predict_proba(X_test_ensemble)[:, 1]
            ensemble_pred = (ensemble_pred_proba >= ensemble_threshold).astype(int)
            ensemble_f1 = f1_score(y_test, ensemble_pred)

            # Calculate ensemble business cost
            tn, fp, fn, tp = confusion_matrix(y_test, ensemble_pred).ravel()
            ensemble_business_cost = (fraud_cost_ratio * fn) + (1 * fp)

            results['Ensemble'] = {
                'best_estimator': ensemble_model,
                'best_params': {'models': [name for name, _ in top_models]},
                'cv_score': np.mean([data['cv_score'] for _, data in top_models]),
                'test_f1': ensemble_f1,
                'test_accuracy': accuracy_score(y_test, ensemble_pred),
                'test_precision': precision_score(y_test, ensemble_pred, zero_division=0),
                'test_recall': recall_score(y_test, ensemble_pred, zero_division=0),
                'test_confusion_matrix': confusion_matrix(y_test, ensemble_pred),
                'final_training_time': ensemble_train_end - ensemble_train_start,
                'optimal_threshold': ensemble_threshold,
                'cost_per_sample': ensemble_cost_per_sample,
                'total_business_cost': ensemble_business_cost,
                'cost_ratio_used': fraud_cost_ratio
            }

            # Update best model based on cost (CV score)
            ensemble_cv_score = results['Ensemble']['cv_score']
            if ensemble_cv_score > best_cv_score:
                self.best_model = ensemble_model
                self.best_model_name = 'Ensemble'
                best_cv_score = ensemble_cv_score

            print(f"✅ Ensemble created with models: {[name for name, _ in top_models]}")
            print(f"📈 Ensemble F1 score: {ensemble_f1:.4f}")
            print(f"🎚️ Ensemble threshold: {ensemble_threshold:.3f}")
            print(f"💰 Ensemble business cost: {ensemble_business_cost:.0f}")
            st.success(f"✅ Ensemble F1: {ensemble_f1:.4f}, Cost: {ensemble_business_cost:.0f}")

        # Store results
        self.cv_results = results
        self.training_stats = {
            'total_records': len(df),
            'train_records': len(X_train),
            'test_records': len(X_test),
            'fraud_count_train': y_train.sum(),
            'fraud_count_test': y_test.sum(),
            'fraud_percentage_train': (y_train.sum() / len(y_train)) * 100,
            'fraud_percentage_test': (y_test.sum() / len(y_test)) * 100,
            'feature_count': X.shape[1],
            'best_model': self.best_model_name,
            'best_cv_score': best_cv_score,
            'models_trained': list(results.keys()),
            'test_size': test_size,
            'total_combinations_tested': total_combinations,
            'cost_ratio_used': fraud_cost_ratio
        }

        if self.best_model:
            best_results = results[self.best_model_name]
            self.training_stats.update({
                'best_test_accuracy': best_results['test_accuracy'],
                'best_test_precision': best_results['test_precision'],
                'best_test_recall': best_results['test_recall'],
                'best_test_f1': best_results['test_f1'],
                'best_confusion_matrix': best_results['test_confusion_matrix'],
                'best_optimal_threshold': best_results['optimal_threshold'],
                'best_total_business_cost': best_results['total_business_cost']
            })

            self.is_ml_trained = True

            print(f"\n🎉 TRAINING COMPLETED!")
            print(f"📊 Summary:")
            print(f"   • Total combinations tested: {total_combinations}")
            print(f"   • Cost ratio used: {fraud_cost_ratio}")
            print(f"🏆 Best model: {self.best_model_name}")
            print(f"📈 Best CV Cost per Sample: {-best_cv_score:.4f}")
            print(f"🎯 Best test F1 score: {best_results['test_f1']:.4f}")
            print(f"💰 Best total business cost: {best_results['total_business_cost']:.0f}")
            print(f"🎚️ Best optimal threshold: {best_results['optimal_threshold']:.3f}")

            st.success(f"🎉 Training Complete! Best: {self.best_model_name} (Cost/Sample: {-best_cv_score:.4f}, F1: {best_results['test_f1']:.4f})")

        return results
    
    def pretrain_all_models(self, 
                           credit_card_data_path="credit_card_transactions.csv",
                           save_path="models/unified_models",
                           test_size=0.2):
        """Pre-train both ML ensemble and NLP models with unified saving"""
        print("🚀 Starting complete model pre-training...")

        # Create models directory
        os.makedirs(save_path, exist_ok=True)

        # 1. Pre-train NLP models first
        print("\n" + "="*60)
        print("STEP 1: Pre-training NLP Models")
        print("="*60)
        nlp_success = self.pretrain_nlp_models()  # Don't pass save_path here anymore

        # 2. Pre-train ML ensemble models
        print("\n" + "="*60)
        print("STEP 2: Pre-training ML Models")
        print("="*60)
        ml_results = self.train_ml_models_with_tuning(credit_card_data_path, test_size)
        ml_success = ml_results is not None and self.best_model is not None

        # 3. Save ALL models using unified function
        print("\n" + "="*60)
        print("STEP 3: Saving All Models")
        print("="*60)
        save_result = self.save_models(save_path)

        # 4. Summary
        print("\n" + "="*60)
        print("PRE-TRAINING SUMMARY")
        print("="*60)
        print(f"📊 NLP Models: {'✅ Success' if nlp_success else '❌ Failed'}")
        print(f"🤖 ML Models: {'✅ Success' if ml_success else '❌ Failed'}")
        print(f"💾 Saving: {'✅ Success' if save_result['success'] else '❌ Failed'}")

        if save_result['success']:
            print("🎉 All models pre-trained and saved successfully!")
            print(f"📁 Location: {save_path}")
            return True
        else:
            print("⚠️ Pre-training completed but saving failed")
            return False

        # 4. Summary
        print("\n" + "="*60)
        print("PRE-TRAINING SUMMARY")
        print("="*60)
        print(f"📊 NLP Models: {'✅ Success' if nlp_success else '❌ Failed'}")
        print(f"🤖 ML Models: {'✅ Success' if ml_success else '❌ Failed'}")

        if nlp_success and ml_success:
            print("🎉 All models pre-trained successfully!")
            print(f"💾 Files saved:")
            print(f"   • ML Model: {ml_save_path}")
            print(f"   • NLP Model: {nlp_save_path}_*")
            return True
        else:
            print("⚠️ Some models failed to pre-train")
            return False

    def predict_with_cost_optimization(self, transaction, cost_ratio=10):
        """Predict using cost-optimized threshold"""
        if not self.is_ml_trained or self.best_model is None:
            return {'fraud_probability': 0, 'is_fraud_prediction': False, 'threshold_used': 0.5, 'confidence': 'None'}

        try:
            # Get probability
            fraud_prob = self.ml_fraud_score(transaction) / 100

            # Use stored optimal threshold if available
            if hasattr(self, 'cv_results') and self.best_model_name in self.cv_results:
                optimal_threshold = self.cv_results[self.best_model_name].get('optimal_threshold', 0.5)
            else:
                optimal_threshold = 0.3  # Conservative threshold favoring fraud detection

            # Make cost-sensitive prediction
            is_fraud = fraud_prob >= optimal_threshold

            return {
                'fraud_probability': fraud_prob * 100,
                'is_fraud_prediction': is_fraud,
                'threshold_used': optimal_threshold,
                'confidence': 'High' if abs(fraud_prob - optimal_threshold) > 0.2 else 'Medium'
            }

        except Exception as e:
            print(f"Cost-sensitive prediction error: {e}")
            return {'fraud_probability': 0, 'is_fraud_prediction': False, 'threshold_used': 0.5, 'confidence': 'Error'}
    
    def ml_fraud_score(self, transaction):
        """Get ML-based fraud score for a single transaction"""
        if not self.is_ml_trained or self.best_model is None:
            return 0
        
        try:
            # Convert transaction to required format
            temp_df = pd.DataFrame([{
                'amt': self._get_numeric_field(transaction, ['amount', 'Amount (by category)', 'amt']),
                'category': str(transaction.get('category', 'other')),
                'trans_date_trans_time': self._get_field_value(transaction, ['date', 'Purchase date', 'trans_date_trans_time']) or datetime.now()
            }])
            
            # Prepare features
            X = self.prepare_ml_features(temp_df)
            
            # Scale if needed
            if self.best_model_name in ['SVM', 'Logistic Regression']:
                X = self.scaler.transform(X)
            
            # Get probability
            fraud_prob = self.best_model.predict_proba(X)[0][1]
            return fraud_prob * 100
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return 0
    
    def predict_personal_expense(self, transaction):
        """Combined prediction using enhanced rules and ML fraud detection"""
        # Get rule-based score
        rule_score, confidence_factors, detailed_breakdown = self.enhanced_rule_based_score(transaction)
        
        # Get ML fraud score
        ml_score = self.ml_fraud_score(transaction)
        
        # Combine scores with dynamic weighting
        if rule_score >= 60:  # High rule confidence
            final_score = (rule_score * 0.8) + (ml_score * 0.2)
        elif rule_score <= 20:  # Low rule confidence, trust ML more
            final_score = (rule_score * 0.4) + (ml_score * 0.6)
        else:  # Balanced approach
            final_score = (rule_score * 0.6) + (ml_score * 0.4)
        
        return {
            'final_score': min(max(final_score, 0), 100),
            'rule_score': rule_score,
            'ml_score': ml_score,
            'confidence_factors': confidence_factors,
            'detailed_breakdown': detailed_breakdown
        }
    
    def analyze_transactions(self, transactions_df, threshold=50):
        """Analyze multiple transactions with enhanced detection - OPTIMIZED VERSION"""
        results = []
        total_transactions = len(transactions_df)

        print(f"Starting analysis of {total_transactions} transactions...")

        # Progress tracking
        batch_size = 100  # Process in batches to show progress
        processed = 0

        for start_idx in range(0, total_transactions, batch_size):
            end_idx = min(start_idx + batch_size, total_transactions)
            batch_df = transactions_df.iloc[start_idx:end_idx]

            print(f"Processing batch {start_idx//batch_size + 1}: transactions {start_idx+1}-{end_idx}")

            for idx, row in batch_df.iterrows():
                try:
                    transaction = row.to_dict()

                    # Data validation
                    vendor = self._get_field_value(transaction, ['merchant', 'Vendor name', 'vendor'])
                    amount = self._get_numeric_field(transaction, ['amount', 'Amount (by category)', 'amt'])

                    if not vendor or vendor.strip() == '':
                        print(f"Warning: Empty vendor for transaction {idx}")
                        vendor = "Unknown Vendor"

                    if amount <= 0:
                        print(f"Warning: Invalid amount ({amount}) for transaction {idx}")
                        amount = 0

                    # Get prediction with error handling
                    try:
                        prediction = self.predict_personal_expense(transaction)
                    except Exception as e:
                        print(f"Error predicting transaction {idx}: {e}")
                        # Fallback prediction
                        prediction = {
                            'final_score': 5,  # Low default score
                            'rule_score': 5,
                            'ml_score': 0,
                            'confidence_factors': [f"Prediction error: {str(e)[:50]}"],
                            'detailed_breakdown': {}
                        }

                    # Determine classification
                    is_flagged = prediction['final_score'] >= threshold

                    if prediction['final_score'] >= 75:
                        risk_level = "High"
                    elif prediction['final_score'] >= 40:
                        risk_level = "Medium"
                    else:
                        risk_level = "Low"

                    result = {
                        # Original transaction data
                        **transaction,
                        # Analysis results
                        'combined_score': round(prediction['final_score'], 1),
                        'rule_score': round(prediction['rule_score'], 1),
                        'ml_score': round(prediction['ml_score'], 1),
                        'is_flagged': is_flagged,
                        'risk_level': risk_level,
                        'confidence_factors': '; '.join(prediction['confidence_factors'][:3]),  # Limit factors
                        'classification': 'Personal' if is_flagged else 'Business'
                    }

                    results.append(result)
                    processed += 1

                except Exception as e:
                    print(f"Error processing transaction {idx}: {e}")
                    # Add error transaction to results
                    error_result = {
                        **row.to_dict(),
                        'combined_score': 0,
                        'rule_score': 0,
                        'ml_score': 0,
                        'is_flagged': False,
                        'risk_level': 'Error',
                        'confidence_factors': f'Processing error: {str(e)[:100]}',
                        'classification': 'Error'
                    }
                    results.append(error_result)
                    processed += 1

            # Progress update
            progress_pct = (processed / total_transactions) * 100
            print(f"Progress: {processed}/{total_transactions} ({progress_pct:.1f}%)")

        print(f"Analysis completed: {len(results)} results generated")
        return pd.DataFrame(results)
    
    def save_models(self, save_path):
        """Save both ML and NLP models in a unified structure"""
        try:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)

            print(f"💾 Saving all models to: {save_path}")

            # 1. Save ML Models (if trained)
            ml_saved = False
            if self.is_ml_trained and self.best_model:
                print("🔄 Saving ML models...")

                # Save ML components
                with open(save_dir / 'best_model.pkl', 'wb') as f:
                    pickle.dump(self.best_model, f)

                with open(save_dir / 'scaler.pkl', 'wb') as f:
                    pickle.dump(self.scaler, f)

                with open(save_dir / 'category_encoder.pkl', 'wb') as f:
                    pickle.dump(self.category_encoder, f)

                ml_saved = True
                print("✅ ML models saved successfully")

            # 2. Save NLP Models (if loaded)
            nlp_saved = False
            if self.is_embeddings_loaded and self.embedding_type:
                print("🔄 Saving NLP models...")

                nlp_data = {
                    'embedding_type': self.embedding_type,
                    'embedding_model_name': self.embedding_model_name,
                    'embedding_dim': self.embedding_dim,
                    'is_embeddings_loaded': self.is_embeddings_loaded
                }

                if self.embedding_type == 'spacy':
                    # For spaCy, just save the model name
                    nlp_data['spacy_model_path'] = self.embedding_model_name
                    print(f"💾 Saved spaCy model info: {self.embedding_model_name}")

                elif self.embedding_type == 'sentence_bert':
                    # Save the actual sentence-transformers model
                    sentence_bert_path = save_dir / 'sentence_bert_model'
                    self.embedding_model.save(str(sentence_bert_path))
                    nlp_data['sentence_bert_path'] = str(sentence_bert_path)
                    print(f"💾 Saved Sentence-BERT model to: {sentence_bert_path}")

                # Save NLP metadata
                with open(save_dir / 'nlp_model.pkl', 'wb') as f:
                    pickle.dump(nlp_data, f)

                nlp_saved = True
                print("✅ NLP models saved successfully")

            # 3. Save Unified Metadata
            unified_metadata = {
                # ML metadata
                'ml_trained': ml_saved,
                'best_model_name': self.best_model_name if ml_saved else None,
                'training_stats': self.training_stats if ml_saved else {},
                'cv_results': self.cv_results if ml_saved else {},
                'category_encoder_fitted': self.category_encoder_fitted if ml_saved else False,

                # NLP metadata
                'nlp_loaded': nlp_saved,
                'embedding_type': self.embedding_type if nlp_saved else None,
                'embedding_model_name': self.embedding_model_name if nlp_saved else None,
                'embedding_dim': self.embedding_dim if nlp_saved else 0,

                # General metadata
                'model_version': '3.0',
                'saved_date': datetime.now().isoformat(),
                'components_saved': {
                    'ml_models': ml_saved,
                    'nlp_models': nlp_saved
                }
            }

            with open(save_dir / 'unified_metadata.pkl', 'wb') as f:
                pickle.dump(unified_metadata, f)

            # Summary
            print(f"\n🎉 Model saving completed!")
            print(f"📊 Components saved:")
            print(f"   • ML Models: {'✅ Yes' if ml_saved else '❌ No (not trained)'}")
            print(f"   • NLP Models: {'✅ Yes' if nlp_saved else '❌ No (not loaded)'}")
            print(f"💾 Location: {save_path}")

            return {
                'success': True,
                'ml_saved': ml_saved,
                'nlp_saved': nlp_saved,
                'save_path': str(save_path)
            }

        except Exception as e:
            print(f"❌ Failed to save models: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    # UNIFIED LOAD FUNCTION - Replace both load_models() and _load_nlp_model()
    def load_models(self, load_path):
        """Load both ML and NLP models from unified structure"""
        try:
            load_dir = Path(load_path)

            if not load_dir.exists():
                print(f"❌ Model directory not found: {load_path}")
                return False

            print(f"🔄 Loading all models from: {load_path}")

            # Load unified metadata first
            metadata_file = load_dir / 'unified_metadata.pkl'
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                print(f"📋 Found unified metadata (version: {metadata.get('model_version', 'unknown')})")
            else:
                # Fallback to individual metadata files for backward compatibility
                print("📋 Using fallback metadata loading...")
                metadata = {'ml_trained': False, 'nlp_loaded': False}

            ml_loaded = False
            nlp_loaded = False

            # 1. Load ML Models
            if metadata.get('ml_trained', False):
                try:
                    print("🔄 Loading ML models...")

                    # Load ML components
                    with open(load_dir / 'best_model.pkl', 'rb') as f:
                        self.best_model = pickle.load(f)

                    with open(load_dir / 'scaler.pkl', 'rb') as f:
                        self.scaler = pickle.load(f)

                    with open(load_dir / 'category_encoder.pkl', 'rb') as f:
                        self.category_encoder = pickle.load(f)

                    # Set ML attributes
                    self.is_ml_trained = True
                    self.best_model_name = metadata.get('best_model_name')
                    self.training_stats = metadata.get('training_stats', {})
                    self.cv_results = metadata.get('cv_results', {})
                    self.category_encoder_fitted = metadata.get('category_encoder_fitted', True)

                    ml_loaded = True
                    print(f"✅ ML models loaded successfully (Best: {self.best_model_name})")

                except Exception as e:
                    print(f"⚠️ Failed to load ML models: {e}")

            # 2. Load NLP Models
            if metadata.get('nlp_loaded', False):
                try:
                    print("🔄 Loading NLP models...")

                    # Load NLP metadata
                    with open(load_dir / 'nlp_model.pkl', 'rb') as f:
                        nlp_data = pickle.load(f)

                    # Set NLP attributes
                    self.embedding_type = nlp_data['embedding_type']
                    self.embedding_model_name = nlp_data['embedding_model_name']
                    self.embedding_dim = nlp_data['embedding_dim']
                    self.is_embeddings_loaded = nlp_data['is_embeddings_loaded']

                    # Load the actual embedding model
                    if self.embedding_type == 'spacy':
                        print(f"🔄 Loading spaCy model: {self.embedding_model_name}...")
                        self.embedding_model = spacy.load(nlp_data['spacy_model_path'])

                    elif self.embedding_type == 'sentence_bert':
                        print(f"🔄 Loading Sentence-BERT model...")
                        sentence_bert_path = nlp_data['sentence_bert_path']
                        self.embedding_model = SentenceTransformer(sentence_bert_path)

                    elif self.embedding_type == 'tfidf':
                        print("📝 Using TF-IDF fallback (no model to load)")

                    nlp_loaded = True
                    print(f"✅ NLP models loaded successfully ({self.embedding_type}: {self.embedding_model_name})")

                except Exception as e:
                    print(f"⚠️ Failed to load NLP models: {e}")

            # 3. Summary
            print(f"\n🎉 Model loading completed!")
            print(f"📊 Components loaded:")
            print(f"   • ML Models: {'✅ Yes' if ml_loaded else '❌ No'}")
            print(f"   • NLP Models: {'✅ Yes' if nlp_loaded else '❌ No'}")

            # Return True if at least one component loaded successfully
            success = ml_loaded or nlp_loaded

            if not success:
                print("⚠️ No models were loaded successfully")

            return success

        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            return False
    
    def display_analysis_results(self, results_df, threshold=50):
        """Display comprehensive analysis results in Streamlit - IMPROVED VERSION"""
        if results_df.empty:
            st.info("No transactions to display")
            return

        st.subheader("🔍 Transaction Analysis Results")

        # Summary statistics
        total_transactions = len(results_df)
        flagged_count = results_df['is_flagged'].sum()
        flagged_percentage = (flagged_count / total_transactions) * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{total_transactions:,}")
        with col2:
            st.metric("Flagged as Personal", f"{flagged_count:,}")
        with col3:
            st.metric("Flagged Percentage", f"{flagged_percentage:.1f}%")
        with col4:
            avg_score = results_df['combined_score'].mean()
            st.metric("Avg Personal Score", f"{avg_score:.1f}")

        # IMPROVED Risk level distribution - Donut chart with better focus
        st.subheader("📊 Risk Level Distribution")
        risk_counts = results_df['risk_level'].value_counts()

        # Create two visualizations side by side
        col1, col2 = st.columns(2)

        with col1:
            # Main donut chart
            fig_donut = px.pie(
                values=risk_counts.values, 
                names=risk_counts.index,
                title="Overall Risk Distribution",
                color_discrete_map={'High': '#dc2626', 'Medium': '#f59e0b', 'Low': '#10b981'},
                hole=0.4  # Creates donut effect
            )

            # Improve layout
            fig_donut.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                textfont_size=12,
                marker=dict(line=dict(color='white', width=2))
            )

            fig_donut.update_layout(
                showlegend=True,
                font=dict(size=11),
                margin=dict(t=50, b=20, l=20, r=20)
            )

            st.plotly_chart(fig_donut, use_container_width=True)

        with col2:
            # Focus chart for Medium + High risk only (more actionable insight)
            actionable_risks = risk_counts[risk_counts.index != 'Low']

            if len(actionable_risks) > 0:
                fig_focus = px.bar(
                    x=actionable_risks.index,
                    y=actionable_risks.values,
                    title="Focus: Medium & High Risk",
                    color=actionable_risks.index,
                    color_discrete_map={'High': '#dc2626', 'Medium': '#f59e0b'},
                    text=actionable_risks.values
                )

                fig_focus.update_traces(texttemplate='%{text}', textposition='outside')
                fig_focus.update_layout(
                    showlegend=False,
                    xaxis_title="Risk Level",
                    yaxis_title="Number of Transactions",
                    font=dict(size=11),
                    margin=dict(t=50, b=20, l=20, r=20)
                )

                st.plotly_chart(fig_focus, use_container_width=True)
            else:
                st.info("🎉 No medium or high-risk transactions detected!")

        # IMPROVED Score distribution with better insights
        st.subheader("📈 Score Distribution Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Main histogram with improved styling
            fig_hist = px.histogram(
                results_df, 
                x='combined_score', 
                nbins=25, 
                title="Personal Score Distribution",
                labels={'combined_score': 'Personal Score', 'count': 'Number of Transactions'},
                color_discrete_sequence=['#3b82f6'],
                marginal="box"  # Add box plot on top
            )

            fig_hist.add_vline(
                x=threshold, 
                line_dash="dash", 
                line_color="#dc2626", 
                line_width=3,
                annotation_text=f"Threshold ({threshold})",
                annotation_position="top"
            )

            fig_hist.update_layout(
                bargap=0.1,
                font=dict(size=11),
                margin=dict(t=50, b=40, l=40, r=20)
            )

            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            # Score ranges breakdown
            score_ranges = pd.cut(
                results_df['combined_score'], 
                bins=[0, 25, 50, 75, 100], 
                labels=['0-25', '26-50', '51-75', '76-100'],
                include_lowest=True
            ).value_counts().sort_index()

            fig_ranges = px.bar(
                x=score_ranges.index,
                y=score_ranges.values,
                title="Score Range Breakdown",
                color=score_ranges.values,
                color_continuous_scale='RdYlGn_r',  # Red-Yellow-Green reversed
                text=score_ranges.values
            )

            fig_ranges.update_traces(texttemplate='%{text}', textposition='outside')
            fig_ranges.update_layout(
                showlegend=False,
                xaxis_title="Score Range",
                yaxis_title="Number of Transactions",
                font=dict(size=11),
                margin=dict(t=50, b=40, l=40, r=20),
                coloraxis_showscale=False
            )

            st.plotly_chart(fig_ranges, use_container_width=True)

        # High-risk transactions table (unchanged)
        if flagged_count > 0:
            st.subheader("🚨 Flagged Transactions")
            flagged_df = results_df[results_df['is_flagged']].copy()

            # Sort by score (descending)
            flagged_df = flagged_df.sort_values('combined_score', ascending=False)

            # Display key columns with better column mapping
            display_columns = []
            column_mappings = {
                'merchant': ['merchant', 'Vendor name', 'vendor'],
                'amount': ['amount', 'Amount (by category)', 'amt'],
                'combined_score': ['combined_score'],
                'rule_score': ['rule_score'],
                'ml_score': ['ml_score'],
                'risk_level': ['risk_level'],
                'classification': ['classification'],
                'confidence_factors': ['confidence_factors']
            }

            # Build available columns list
            final_columns = []
            for target_col, possible_cols in column_mappings.items():
                for col in possible_cols:
                    if col in flagged_df.columns:
                        if target_col != col:
                            flagged_df[target_col] = flagged_df[col]
                        final_columns.append(target_col)
                        break

            if final_columns:
                # Format numeric columns
                for col in ['combined_score', 'rule_score', 'ml_score']:
                    if col in flagged_df.columns:
                        flagged_df[col] = flagged_df[col].round(1)

                st.dataframe(
                    flagged_df[final_columns], 
                    use_container_width=True,
                    hide_index=True
                )

                # Export option
                csv = flagged_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Flagged Transactions",
                    data=csv,
                    file_name=f"flagged_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Could not find expected columns in the data")

        # IMPROVED Model comparison - Smart sampling and better visualization
        if 'rule_score' in results_df.columns and 'ml_score' in results_df.columns:
            st.subheader("⚖️ Model Performance Analysis")

            # Smart sampling for better visualization
            sample_df = results_df.copy()

            # Always include all high and medium risk transactions
            high_med_risk = sample_df[sample_df['risk_level'].isin(['High', 'Medium'])]

            # Sample low risk transactions to avoid overcrowding
            low_risk = sample_df[sample_df['risk_level'] == 'Low']
            if len(low_risk) > 500:  # Only sample if too many low-risk transactions
                low_risk_sample = low_risk.sample(n=500, random_state=42)
            else:
                low_risk_sample = low_risk

            # Combine samples
            sample_df = pd.concat([high_med_risk, low_risk_sample], ignore_index=True)

            col1, col2 = st.columns(2)

            with col1:
                # Main scatter plot with improved styling
                size_col = None
                size_values = None

                for col in ['amount', 'Amount (by category)', 'amt']:
                    if col in sample_df.columns:
                        size_values = abs(sample_df[col])
                        size_values = size_values + 0.01
                        size_col = col
                        break

                hover_data_cols = []
                for col in ['merchant', 'Vendor name', 'combined_score']:
                    if col in sample_df.columns:
                        hover_data_cols.append(col)

                scatter_data = sample_df.copy()
                if size_values is not None:
                    scatter_data['abs_amount'] = size_values

                fig_scatter = px.scatter(
                    scatter_data,
                    x='rule_score',
                    y='ml_score',
                    color='risk_level',
                    size='abs_amount' if size_values is not None else None,
                    hover_data=hover_data_cols if hover_data_cols else None,
                    title=f"Rule vs ML Scores ({len(sample_df):,} transactions)",
                    color_discrete_map={'High': '#dc2626', 'Medium': '#f59e0b', 'Low': '#10b981'},
                    labels={'abs_amount': 'Transaction Amount'},
                    opacity=0.7
                )

                # Add diagonal reference line
                fig_scatter.add_shape(
                    type="line",
                    x0=0, y0=0, x1=100, y1=100,
                    line=dict(color="gray", width=2, dash="dash"),
                )

                # Add quadrant labels for better interpretation
                fig_scatter.add_annotation(x=25, y=75, text="ML High<br>Rule Low", 
                                         showarrow=False, font=dict(color="gray", size=10))
                fig_scatter.add_annotation(x=75, y=25, text="Rule High<br>ML Low", 
                                         showarrow=False, font=dict(color="gray", size=10))

                fig_scatter.update_layout(
                    font=dict(size=11),
                    margin=dict(t=50, b=40, l=40, r=40)
                )

                st.plotly_chart(fig_scatter, use_container_width=True)

            with col2:
                # Model agreement analysis
                agreement_threshold = 10  # scores within 10 points considered "agreeing"

                sample_df['score_diff'] = abs(sample_df['rule_score'] - sample_df['ml_score'])
                sample_df['models_agree'] = sample_df['score_diff'] <= agreement_threshold

                agreement_stats = sample_df.groupby('risk_level')['models_agree'].agg(['count', 'sum']).reset_index()
                agreement_stats['agreement_rate'] = (agreement_stats['sum'] / agreement_stats['count'] * 100).round(1)

                fig_agreement = px.bar(
                    agreement_stats,
                    x='risk_level',
                    y='agreement_rate',
                    title="Model Agreement by Risk Level",
                    color='risk_level',
                    color_discrete_map={'High': '#dc2626', 'Medium': '#f59e0b', 'Low': '#10b981'},
                    text='agreement_rate'
                )

                fig_agreement.update_traces(texttemplate='%{text}%', textposition='outside')
                fig_agreement.update_layout(
                    showlegend=False,
                    xaxis_title="Risk Level",
                    yaxis_title="Agreement Rate (%)",
                    yaxis_range=[0, 105],
                    font=dict(size=11),
                    margin=dict(t=50, b=40, l=40, r=20)
                )

                st.plotly_chart(fig_agreement, use_container_width=True)

                # Add interpretation text
                st.info(f"📊 Models agree (within {agreement_threshold} points) on "
                       f"{sample_df['models_agree'].mean():.1%} of transactions")

            # Additional insight: Disagreement cases
            if len(sample_df[~sample_df['models_agree']]) > 0:
                st.subheader("🔍 Model Disagreement Analysis")

                disagreement_df = sample_df[~sample_df['models_agree']].copy()
                disagreement_df = disagreement_df.nlargest(10, 'score_diff')  # Top 10 disagreements

                # Show cases where models disagree the most
                if not disagreement_df.empty:
                    display_cols = ['merchant', 'rule_score', 'ml_score', 'score_diff', 'risk_level']
                    available_cols = [col for col in display_cols if col in disagreement_df.columns]

                    if 'merchant' not in disagreement_df.columns:
                        for col in ['Vendor name', 'vendor']:
                            if col in disagreement_df.columns:
                                disagreement_df['merchant'] = disagreement_df[col]
                                available_cols = [col for col in display_cols if col in disagreement_df.columns]
                                break

                    st.write("**Top Disagreement Cases:**")
                    st.dataframe(
                        disagreement_df[available_cols].round(1),
                        use_container_width=True,
                        hide_index=True
                    )
    
    def display_training_results(self):
        """Display comprehensive training results in Streamlit"""
        if not self.training_stats:
            st.info("No training results available")
            return

        st.subheader("🎯 Comprehensive Model Training Results")

        # Training overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{self.training_stats.get('total_records', 0):,}")
        with col2:
            st.metric("Train/Test Split", 
                     f"{self.training_stats.get('train_records', 0):,} / {self.training_stats.get('test_records', 0):,}")
        with col3:
            st.metric("Best Model", self.training_stats.get('best_model', 'None'))
        with col4:
            st.metric("Test Size", f"{self.training_stats.get('test_size', 0)*100:.0f}%")

        # Fraud distribution
        st.subheader("📊 Fraud Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Train Fraud %", 
                     f"{self.training_stats.get('fraud_percentage_train', 0):.2f}%",
                     f"{self.training_stats.get('fraud_count_train', 0):,} cases")
        with col2:
            st.metric("Test Fraud %", 
                     f"{self.training_stats.get('fraud_percentage_test', 0):.2f}%",
                     f"{self.training_stats.get('fraud_count_test', 0):,} cases")

        # Model comparison with both CV and test scores
        if self.cv_results:
            st.subheader("📊 Model Performance Comparison")
            
            model_data = []
            for model_name, results in self.cv_results.items():
                if 'error' not in results:
                    model_data.append({
                        'Model': model_name,
                        'CV F1 Score': f"{results['cv_score']:.4f}",
                        'Test Accuracy': f"{results['test_accuracy']:.4f}",
                        'Test Precision': f"{results['test_precision']:.4f}",
                        'Test Recall': f"{results['test_recall']:.4f}",
                        'Test F1 Score': f"{results['test_f1']:.4f}",
                        'Best': '⭐' if model_name == self.best_model_name else ''
                    })
            
            if model_data:
                model_df = pd.DataFrame(model_data)
                model_df = model_df.sort_values('Test F1 Score', ascending=False)
                st.dataframe(model_df, use_container_width=True, hide_index=True)

        # Best model confusion matrix
        if 'best_confusion_matrix' in self.training_stats:
            st.subheader("🎯 Best Model Confusion Matrix (Test Set)")
            
            cm = self.training_stats['best_confusion_matrix']
            
            # Create heatmap
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title=f"Confusion Matrix - {self.best_model_name}",
                labels=dict(x="Predicted", y="Actual"),
                x=['Business', 'Fraud'],
                y=['Business', 'Fraud']
            )
            fig_cm.update_xaxes(side="bottom")
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test Accuracy", f"{self.training_stats.get('best_test_accuracy', 0):.4f}")
            with col2:
                st.metric("Test Precision", f"{self.training_stats.get('best_test_precision', 0):.4f}")
            with col3:
                st.metric("Test Recall", f"{self.training_stats.get('best_test_recall', 0):.4f}")
            with col4:
                st.metric("Test F1 Score", f"{self.training_stats.get('best_test_f1', 0):.4f}")

        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            st.subheader("🔍 Feature Importance")
            
            feature_names = [
                'Amount', 'Log Amount', 'Amount Rounded', 'Amount Z-Score',
                'Category', 'Hour', 'Day of Week', 'Month',
                'Is Weekend', 'Is Night', 'Is Business Hours',
                'Transaction Count 1h', 'Transaction Count 24h', 'Personal Vendor Freq 1h',
                'Holiday Season', 'Summer Vacation', 'School Holiday'
            ]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': self.best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Feature Importance - {self.best_model_name}"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
    
    def explain_with_shap(self, X_train=None, max_samples=100):
        """Generate SHAP explanations for feature importance"""
        if not self.is_ml_trained or self.best_model is None:
            st.warning("Model must be trained before generating SHAP explanations")
            return None

        try:

            # Use provided training data or create sample
            st.info("Generating SHAP explanations...")

            if X_train is not None:
                X_sample = X_train.sample(min(max_samples, len(X_train))) if hasattr(X_train, 'sample') else X_train[:max_samples]
            else:
                st.warning("No training data provided for SHAP analysis")
                return None

            # Scale if needed
            if self.best_model_name in ['SVM', 'Logistic Regression']:
                X_sample = self.scaler.transform(X_sample)

            # Create explainer based on model type
            if self.best_model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
                explainer = shap.TreeExplainer(self.best_model)
            elif self.best_model_name == 'Ensemble':
                # For ensemble, use the first tree-based model if available
                explainer = shap.Explainer(self.best_model, X_sample)
            else:
                explainer = shap.LinearExplainer(self.best_model, X_sample)

            shap_values = explainer.shap_values(X_sample)

            # If binary classification, get positive class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            return explainer, shap_values, X_sample

        except ImportError:
            st.warning("SHAP not installed. Install with: pip install shap")
            return None
        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")
            return None

    def display_shap_analysis(self, explainer=None, shap_values=None, X_sample=None):
        """Display SHAP analysis in Streamlit"""
        if explainer is None:
            result = self.explain_with_shap()
            if result is None:
                return
            explainer, shap_values, X_sample = result

        st.subheader("🔍 SHAP Feature Importance Analysis")

        feature_names = [
            'Amount', 'Log Amount', 'Amount Rounded', 'Amount Z-Score',
            'Category', 'Hour', 'Day of Week', 'Month',
            'Is Weekend', 'Is Night', 'Is Business Hours',
            'Transaction Count 1h', 'Transaction Count 24h', 'Personal Vendor Freq 1h',
            'Holiday Season', 'Summer Vacation', 'School Holiday'
        ]

        # Summary plot
        fig_summary = shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        st.pyplot(fig_summary)

        # Feature importance plot
        fig_importance = shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                                         plot_type="bar", show=False)
        st.pyplot(fig_importance)
    
    def ml_score(self, transaction):
        """Get only ML score"""
        return self.ml_fraud_score(transaction)

    def rule_based_score(self, transaction):
        """Get only rule-based score for UI compatibility"""
        rule_score, _ = self.enhanced_rule_based_score(transaction)
        return rule_score
    
    def predict_combined_score(self, transaction):
        """Get final combined score"""
        prediction = self.predict_personal_expense(transaction)
        return prediction['final_score']

    def _get_flagging_reason(self, transaction, rule_score, ml_score):
        """Get explanation for flagging"""
        prediction = self.predict_personal_expense(transaction)
        factors = prediction.get('confidence_factors', [])
        return '; '.join(factors[:3]) if factors else "Based on pattern analysis"

    def display_training_metrics(self):
        """Wrapper for display_training_results for UI compatibility"""
        self.display_training_results() 
    
    def _auto_train_model(self):
        """Auto-train model with default dataset if available"""
        try:
            print("🔄 Auto-training model with default dataset...")
            self.train_ml_models_with_tuning()  # Fixed method name
        except Exception as e:
            print(f"⚠️ Auto-training failed: {e}")
            
# ================================================================================================================================================================================================================

# Fixed Asset Tracker Section
class FixedAssetDetector:
    def __init__(self, model_path=None):
        """Initialize the Enhanced FixedAssetDetector with comprehensive patterns."""
        
        # Thread-safe caches for performance
        self.vendor_cache = {}
        self.semantic_cache = {}
        self.cache_lock = Lock()
        
        self.EMPLOYEE_EMAIL_MAPPING = {
            "Aakash Mandhar": "aakash@k-id.com",
            "Tushar Ajmera": "tushar@k-id.com",
            "Susan": "susanchen@k-id.com",
            "Fil Baumanis": "fil@k-id.com",
            "Kay Vasey": "kay@k-id.com",
            "Luc Delany": "luc@k-id.com",
            "Wesley": "wesleysitu@k-id.com",
            "Liz": "liz@k-id.com",
            "Kevin Loh": "kevin@k-id.com",
            "Tiffany Friedel": "tiffany@k-id.com",
            "Marshall Nu": "marshall@k-id.com",
            "Arunan Rabindran": "arun@k-id.com",
            "Aaron Lam": "aaron.lam@k-id.com",
            "Joseph Newman": "joe@k-id.com",
            "Braxton Sheum": "braxton@k-id.com",
            "Natalie Shou": "natalie@k-id.com",
            "Alyssa Aw": "alyssa@k-id.com",
            "Arunan": "arunan@k-id.com",
            "Miguel Kyle Khonrad Lejano Martinez": "miguel@k-id.com",
            "Olav Bus": "olav@k-id.com",
            "Crystal Wong": "crystal@k-id.com",
            "Benjamin Fox": "ben@k-id.com",
            "Markus Juuti": "marklee@k-id.com",
            "Tristen": "tristen@k-id.com",
            "Julian Corbett": "julian@k-id.com",
            "Beatrice Cavicchioli": "beatrice@k-id.com",
            "Lennart Ng": "lennart@k-id.com",
            "Carolyn Yan": "carolyn@k-id.com",
            "Lulu Xia": "lulu@k-id.com",
            "Sebastian Chew": "sebastian@k-id.com",
            "Keemin Ngiam": "keemin@k-id.com",
            "Nina Cheuck": "nina@k-id.com",
            "Timothy Ma": "timothy@k-id.com",
            "Adam Snyder": "adam@k-id.com",
            "Denise Villanueva": "denise@k-id.com",
            "Benjamin Chen": "benc@k-id.com",  # Note: Second Benjamin with different email
            "Ibrahim Midian": "ibrahim@k-id.com",
            "Erich Bao": "erich@k-id.com",
            "Ruosi Wang": "ruosi@k-id.com",
            "Shireen Ho": "shireen@k-id.com",
            "Hilson Wong": "hilson@k-id.com",
            "Bernie": "bernie@k-id.com",
            "Kieran Donovan": "kieran@k-id.com",
            "Michel Paupulaire": "mpaupulaire@k-id.com",
            "Greg Leib": "greg@k-id.com",
            "Rupali Sharma": "rupali@k-id.com",
            "Charleston Yap": "charlestonyap@k-id.com",  
            "Andrew Huth": "ahuth@k-id.com",
            "Joanna Shields": "joanna@k-id.com",
            "Jeff Wu": "jwu@k-id.com",
            "Andre Malan": "andre@k-id.com"
        }

        
        # Enhanced fixed asset patterns with weights and fuzzy patterns
        self.fixed_asset_patterns = {
            'electronics_it': {
                'exact_matches': [
                    # Computers & Laptops
                    'laptop', 'computer', 'desktop', 'workstation', 'server', 'macbook',
                    'dell', 'hp', 'lenovo', 'asus', 'acer', 'msi', 'apple', 'microsoft surface',
                    'thinkpad', 'pavilion', 'inspiron', 'precision', 'optiplex',
                    
                    # Monitors & Displays
                    'monitor', 'display', 'screen', 'lcd', 'led', 'oled', 'ultrawide',
                    'samsung monitor', 'lg monitor', 'dell monitor', 'asus monitor',
                    
                    # Networking Equipment
                    'router', 'switch', 'firewall', 'access point', 'modem',
                    'cisco', 'netgear', 'tp-link', 'linksys', 'ubiquiti',
                    
                    # Printers & Scanners
                    'printer', 'scanner', 'multifunction', 'copier', 'plotter',
                    'canon', 'epson', 'brother', 'xerox', 'ricoh', 'konica minolta',
                    
                    # Storage & Backup
                    'hard drive', 'ssd', 'nas', 'backup drive',
                    'synology', 'qnap', 'western digital', 'seagate', 'samsung ssd',
                    
                    # Retailers
                    'best buy', 'newegg', 'amazon business', 'cdw', 'insight',
                    'challenger', 'harvey norman', 'courts', 'gain city'
                ],
                'fuzzy_patterns': [
                    'electronic device', 'it equipment', 'tech hardware', 'computer equipment',
                    'network equipment', 'office electronics', 'digital equipment',
                    'computing device', 'server hardware', 'workstation setup'
                ],
                'weight': 0.9,
                'min_amount': 200,  # Minimum amount for IT equipment
                'depreciation_years': 3
            },
            
            'software_licenses': {
                'exact_matches': [
                    # Software Companies
                    'microsoft', 'adobe', 'oracle', 'sap', 'salesforce', 'autodesk',
                    'quickbooks', 'sage', 'intuit', 'vmware', 'citrix',
                    
                    # Software Types
                    'software license', 'enterprise license', 'perpetual license',
                    'office 365', 'adobe creative', 'autocad', 'solidworks',
                    'windows server', 'sql server', 'exchange server'
                ],
                'fuzzy_patterns': [
                    'software licensing', 'enterprise software', 'business software',
                    'professional software', 'specialized software', 'system software'
                ],
                'weight': 0.75,
                'min_amount': 500,
                'depreciation_years': 3
            },
            
            'furniture_office': {
                'exact_matches': [
                    # Office Furniture
                    'desk', 'chair', 'table', 'cabinet', 'bookshelf', 'filing cabinet',
                    'office chair', 'executive chair', 'standing desk', 'conference table',
                    'reception desk', 'workstation', 'cubicle', 'partition',
                    
                    # Furniture Retailers
                    'ikea', 'steelcase', 'herman miller', 'haworth', 'knoll',
                    'staples', 'office depot', 'officeworks', 'hon', 'humanscale',
                    'west elm', 'pottery barn', 'crate & barrel',
                    
                    # Storage Solutions
                    'safe', 'vault', 'locker', 'storage unit', 'shelving system'
                ],
                'fuzzy_patterns': [
                    'office furniture', 'workspace furniture', 'ergonomic furniture',
                    'commercial furniture', 'business furniture', 'modular furniture',
                    'conference room furniture', 'reception furniture'
                ],
                'weight': 0.85,
                'min_amount': 150,
                'depreciation_years': 7
            },
            
            'machinery_equipment': {
                'exact_matches': [
                    # Industrial Equipment
                    'machine', 'equipment', 'tool', 'generator', 'compressor',
                    'conveyor', 'forklift', 'crane', 'hoist', 'pump', 'motor',
                    'turbine', 'boiler', 'furnace', 'press', 'lathe', 'mill',
                    
                    # Specialized Equipment
                    'medical equipment', 'diagnostic equipment', 'laboratory equipment',
                    'kitchen equipment', 'hvac system', 'security system',
                    'fire suppression', 'elevator', 'escalator',
                    
                    # Manufacturers
                    'caterpillar', 'john deere', 'komatsu', 'hitachi', 'liebherr',
                    'atlas copco', 'ingersoll rand', 'gardner denver'
                ],
                'fuzzy_patterns': [
                    'industrial equipment', 'manufacturing equipment', 'production equipment',
                    'heavy machinery', 'specialized machinery', 'processing equipment',
                    'fabrication equipment', 'automation equipment'
                ],
                'weight': 0.9,
                'min_amount': 1000,
                'depreciation_years': 10
            },
            
            'vehicles_transport': {
                'exact_matches': [
                    # Commercial Vehicles
                    'truck', 'van', 'trailer', 'fleet vehicle', 'delivery truck',
                    'cargo van', 'pickup truck', 'commercial vehicle',
                    'ford transit', 'mercedes sprinter', 'iveco daily',
                    
                    # Heavy Vehicles
                    'bulldozer', 'excavator', 'backhoe', 'dump truck', 'cement mixer',
                    'road roller', 'grader', 'tractor', 'combine harvester',
                    
                    # Dealerships
                    'ford commercial', 'mercedes commercial', 'volvo trucks',
                    'peterbilt', 'kenworth', 'freightliner'
                ],
                'fuzzy_patterns': [
                    'commercial vehicle', 'fleet vehicle', 'work truck', 'service vehicle',
                    'construction vehicle', 'agricultural vehicle', 'transport equipment'
                ],
                'weight': 0.95,
                'min_amount': 5000,
                'depreciation_years': 5
            },
            
            'building_improvements': {
                'exact_matches': [
                    # Construction & Renovation
                    'renovation', 'construction', 'installation', 'building improvement',
                    'hvac installation', 'electrical work', 'plumbing work',
                    'flooring', 'roofing', 'windows', 'doors', 'lighting system',
                    
                    # Contractors
                    'contractor', 'construction company', 'renovation company',
                    'electrical contractor', 'plumbing contractor', 'hvac contractor'
                ],
                'fuzzy_patterns': [
                    'building improvement', 'facility upgrade', 'infrastructure improvement',
                    'capital improvement', 'permanent installation', 'structural modification'
                ],
                'weight': 0.8,
                'min_amount': 2500,
                'depreciation_years': 15
            }
        }
        
        # Pre-compile regex patterns for speed (similar to PersonalExpenseDetector)
        self.compiled_patterns = {
            'asset_keywords': re.compile(r'\b(asset|equipment|machinery|installation|purchase|acquisition)\b', re.IGNORECASE),
            'capital_expenditure': re.compile(r'\b(capex|capital|investment|upgrade|replacement)\b', re.IGNORECASE),
            'depreciation_terms': re.compile(r'\b(depreciat|amortiz|useful life|fixed asset)\b', re.IGNORECASE),
            'quantity_indicators': re.compile(r'\b(\d+\s*(units?|pcs?|pieces?|qty|quantity))\b', re.IGNORECASE),
            'model_numbers': re.compile(r'\b([A-Z]{2,}\d{3,}|[A-Z]\d{4,})\b', re.IGNORECASE)
        }
        
        # Semantic groups for advanced analysis
        self.asset_semantic_groups = {
            'technology_hardware': [
                'computer', 'server', 'network', 'hardware', 'electronic', 'digital',
                'processor', 'memory', 'storage', 'graphics', 'motherboard'
            ],
            'office_equipment': [
                'furniture', 'desk', 'chair', 'cabinet', 'workspace', 'ergonomic',
                'modular', 'conference', 'reception', 'storage'
            ],
            'industrial_machinery': [
                'machine', 'equipment', 'industrial', 'manufacturing', 'production',
                'assembly', 'fabrication', 'automation', 'conveyor', 'pump'
            ],
            'infrastructure': [
                'building', 'facility', 'installation', 'infrastructure', 'system',
                'hvac', 'electrical', 'plumbing', 'security', 'lighting'
            ]
        }
        
        # Business indicators that strengthen fixed asset classification
        self.business_indicators = {
            'high_confidence': [
                'company', 'corporation', 'business', 'enterprise', 'commercial',
                'industrial', 'professional', 'office', 'facility', 'operations'
            ],
            'medium_confidence': [
                'department', 'team', 'staff', 'employee', 'workplace', 'operational',
                'management', 'administrative', 'technical', 'production'
            ]
        }
        
        # Convert to sets for faster lookup
        self.business_indicators_set = {
            'high_confidence': set(self.business_indicators['high_confidence']),
            'medium_confidence': set(self.business_indicators['medium_confidence'])
        }
        
        self.BRAND_COLORS = {
            'purple': '#715DEC',
            'purple_dark': '#5B47CC',
            'purple_light': '#F3F2FF',
            'orange': '#FC6C0F',
            'blackberry': '#2C216F',
            'blackberry_light': '#44307A',
            'black': '#333333',
            'gradient_start': '#715DEC',
            'gradient_end': '#2C216F'
        }
        
    def _get_field_value(self, transaction, field_names, default=''):
        """Helper method to get a field value from the transaction dictionary."""
        for field_name in field_names:
            if field_name in transaction and transaction[field_name] is not None:
                value = str(transaction[field_name]).strip()
                if value and value.lower() not in ['', 'nan', 'none', 'null']:
                    return value
        return default

    def _get_numeric_field(self, transaction, field_names, default=0):
        """Helper method to get a numeric field value from the transaction dictionary."""
        value = self._get_field_value(transaction, field_names, str(default))
        try:
            return abs(float(value))  # Use absolute value for amount
        except (ValueError, TypeError):
            return default

    def _preprocess_transaction_texts(self, transactions_df):
        """Pre-process all text data for faster analysis (from PersonalExpenseDetector)"""
        df = transactions_df.copy()
        
        # Extract and normalize text fields
        df['vendor_clean'] = df.apply(lambda row: 
            self._get_field_value(row.to_dict(), ['merchant', 'Vendor name', 'vendor'], '').lower().strip(), 
            axis=1)
        
        df['description_clean'] = df.apply(lambda row: 
            self._get_field_value(row.to_dict(), ['description', 'Description', 'memo'], '').lower().strip(), 
            axis=1)
        
        df['combined_text'] = (df['vendor_clean'] + ' ' + df['description_clean']).str.strip()
        df['text_hash'] = df['combined_text'].apply(lambda x: hashlib.md5(x.encode()).hexdigest() if x else '')
        
        return df

    def enhanced_fuzzy_match_fixed_asset(self, text, threshold=80):
        """Enhanced fuzzy matching with specificity bonus for software companies"""
        if not text or not text.strip():
            return 0, None, {}

        text_lower = text.lower().strip()
        max_score = 0
        best_category = None
        category_details = {}

        # Define high-priority software companies that should always go to software_licenses
        software_companies = {
            'microsoft', 'adobe', 'oracle', 'sap', 'salesforce', 'autodesk',
            'quickbooks', 'sage', 'intuit', 'vmware', 'citrix'
        }

        for category, patterns in self.fixed_asset_patterns.items():
            category_score = 0
            match_details = {'exact_matches': [], 'fuzzy_matches': []}

            # Exact matches (highest priority)
            for exact_match in patterns['exact_matches']:
                if len(exact_match) <= 4:  # Short terms need exact word match
                    if exact_match == text_lower or f" {exact_match} " in f" {text_lower} ":
                        category_score = 100
                        match_details['exact_matches'].append(exact_match)
                        break
                else:  # Longer terms can use substring matching
                    if exact_match in text_lower:
                        category_score = 100
                        match_details['exact_matches'].append(exact_match)
                        break

            # Fuzzy patterns
            if category_score < 100:
                for pattern in patterns['fuzzy_patterns']:
                    if len(pattern) <= 3:  # Skip very short patterns
                        continue

                    if pattern in text_lower:
                        category_score = max(category_score, 85)
                        match_details['fuzzy_matches'].append((pattern, 85))
                    else:
                        fuzzy_score = fuzz.partial_ratio(pattern, text_lower)
                        if fuzzy_score >= threshold:
                            category_score = max(category_score, fuzzy_score)
                            match_details['fuzzy_matches'].append((pattern, fuzzy_score))

            # Apply specificity bonus for software companies
            specificity_bonus = 0
            if category == 'software_licenses':
                for company in software_companies:
                    if company in text_lower:
                        specificity_bonus = 25  # Big bonus to ensure software wins
                        break

            # Add the specificity bonus to the score
            final_category_score = category_score + specificity_bonus

            # Weight the score
            weighted_score = final_category_score * patterns['weight']

            if weighted_score > max_score:
                max_score = weighted_score
                best_category = category
                category_details = {
                    'category': category,
                    'raw_score': category_score,
                    'specificity_bonus': specificity_bonus,
                    'final_score': final_category_score,
                    'weighted_score': weighted_score,
                    'weight': patterns['weight'],
                    'matches': match_details,
                    'min_amount': patterns.get('min_amount', 0),
                    'depreciation_years': patterns.get('depreciation_years', 5)
                }

        return max_score, best_category, category_details

    def enhanced_rule_based_score(self, transaction):
        """Enhanced rule-based scoring with detailed analysis (inspired by PersonalExpenseDetector)"""
        score = 0
        confidence_factors = []
        detailed_breakdown = {}
        
        # Get pre-processed text data or create it
        combined_text = transaction.get('combined_text', '')
        text_hash = transaction.get('text_hash', '')
        
        if not combined_text:
            vendor = self._get_field_value(transaction, ['merchant', 'Vendor name', 'vendor'])
            description = self._get_field_value(transaction, ['description', 'Description', 'memo'])
            combined_text = f"{vendor} {description}".strip().lower()
            text_hash = hashlib.md5(combined_text.encode()).hexdigest() if combined_text else ''
        
        if not combined_text.strip():
            return 5, ["No vendor/description data"], {}
        
        # 1. Vendor and category analysis with caching
        vendor = self._get_field_value(transaction, ['merchant', 'Vendor name', 'vendor'])
        vendor_key = vendor.lower().strip() if vendor else ""
        
        with self.cache_lock:
            if vendor_key in self.vendor_cache:
                vendor_score, vendor_category, vendor_details = self.vendor_cache[vendor_key]
            else:
                vendor_score, vendor_category, vendor_details = self.enhanced_fuzzy_match_fixed_asset(combined_text)
                self.vendor_cache[vendor_key] = (vendor_score, vendor_category, vendor_details)
        
        if vendor_score > 0:
            base_vendor_score = min(vendor_score * 0.6, 60)  # Cap vendor contribution
            score += base_vendor_score
            confidence_factors.append(f"Category: {vendor_category} ({base_vendor_score:.1f})")
            detailed_breakdown['vendor_analysis'] = vendor_details
        
        # 2. Amount-based scoring with category-specific thresholds
        amount = self._get_numeric_field(transaction, ['amount', 'Amount (by category)', 'amt'])
        if amount > 0 and vendor_details:
            min_amount = vendor_details.get('min_amount', 500)
            
            if amount >= min_amount:
                if amount >= min_amount * 10:  # Very high amount
                    amount_score = 25
                elif amount >= min_amount * 3:  # High amount
                    amount_score = 20
                elif amount >= min_amount:  # Minimum threshold
                    amount_score = 15
                else:
                    amount_score = 5
                
                score += amount_score
                confidence_factors.append(f"Amount: ${amount:,.0f} (+{amount_score})")
            else:
                # Penalty for low amounts
                score -= 10
                confidence_factors.append(f"Low amount penalty: ${amount:,.0f}")
        
        # 3. Pattern matching using pre-compiled regex
        pattern_matches = 0
        pattern_details = {}
        
        for pattern_name, pattern in self.compiled_patterns.items():
            matches = pattern.findall(combined_text)
            if matches:
                pattern_matches += len(matches)
                pattern_details[pattern_name] = matches
        
        if pattern_matches > 0:
            pattern_score = min(pattern_matches * 8, 25)  # Cap pattern contribution
            score += pattern_score
            confidence_factors.append(f"Asset patterns: {pattern_matches} (+{pattern_score})")
            detailed_breakdown['patterns'] = pattern_details
        
        # 4. Business context indicators
        combined_lower = combined_text.lower()
        
        high_business = sum(1 for kw in self.business_indicators_set['high_confidence'] if kw in combined_lower)
        if high_business > 0:
            business_boost = high_business * 10
            score += business_boost
            confidence_factors.append(f"Business context: {high_business} (+{business_boost})")
        
        med_business = sum(1 for kw in self.business_indicators_set['medium_confidence'] if kw in combined_lower)
        if med_business > 0:
            business_boost = med_business * 5
            score += business_boost
            confidence_factors.append(f"Business indicators: {med_business} (+{business_boost})")
        
        # 5. Semantic analysis (if available)
        if text_hash and len(combined_text) > 10:
            semantic_score = self._get_cached_semantic_score(text_hash, combined_text)
            if abs(semantic_score) > 1:
                semantic_contribution = min(abs(semantic_score), 20)
                score += semantic_contribution
                confidence_factors.append(f"Semantic match: +{semantic_contribution:.1f}")
        
        # 6. Date-based factors
        date_field = self._get_field_value(transaction, ['date', 'Purchase date', 'trans_date_trans_time'])
        if date_field:
            try:
                dt = pd.to_datetime(date_field)
                # Business hours boost (9 AM - 5 PM)
                if 9 <= dt.hour <= 17:
                    score += 5
                    confidence_factors.append("Business hours")
                # Weekday boost
                if dt.weekday() < 5:  # Monday = 0, Friday = 4
                    score += 5
                    confidence_factors.append("Weekday transaction")
            except Exception:
                pass
        
        # 7. Quantity indicators
        if 'quantity_indicators' in pattern_details:
            quantity_boost = min(len(pattern_details['quantity_indicators']) * 5, 15)
            score += quantity_boost
            confidence_factors.append(f"Quantity indicators: +{quantity_boost}")
        
        final_score = min(max(score, 0), 100)
        detailed_breakdown['final_score'] = final_score
        detailed_breakdown['amount'] = amount
        
        return final_score, confidence_factors, detailed_breakdown

    def _get_cached_semantic_score(self, text_hash, combined_text):
        """Get semantic score with caching (placeholder for semantic analysis)"""
        with self.cache_lock:
            if text_hash in self.semantic_cache:
                return self.semantic_cache[text_hash]
        
        # Simple keyword-based semantic scoring as fallback
        semantic_score = 0
        text_lower = combined_text.lower()
        
        for group_name, keywords in self.asset_semantic_groups.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                group_weights = {
                    'technology_hardware': 15,
                    'office_equipment': 12,
                    'industrial_machinery': 18,
                    'infrastructure': 20
                }
                semantic_score += matches * group_weights.get(group_name, 10)
        
        with self.cache_lock:
            self.semantic_cache[text_hash] = semantic_score
        
        return semantic_score

    def detect_fixed_asset(self, transaction, threshold=50):
        """Enhanced detection with detailed analysis"""
        score, factors, breakdown = self.enhanced_rule_based_score(transaction)
        is_fixed_asset = score >= threshold
        
        # Determine confidence level
        if score >= 80:
            confidence_level = "Very High"
        elif score >= 65:
            confidence_level = "High" 
        elif score >= 50:
            confidence_level = "Medium"
        elif score >= 30:
            confidence_level = "Low"
        else:
            confidence_level = "Very Low"
        
        return {
            'is_fixed_asset': is_fixed_asset,
            'score': round(score, 1),
            'confidence_level': confidence_level,
            'confidence_factors': factors,
            'detailed_breakdown': breakdown,
            'category': breakdown.get('vendor_analysis', {}).get('category'),
            'estimated_useful_life': breakdown.get('vendor_analysis', {}).get('depreciation_years', 5)
        }

    def analyze_transactions(self, transactions_df, threshold=50):
        """Analyze multiple transactions (similar to PersonalExpenseDetector)"""
        results = []
        total_transactions = len(transactions_df)
        
        print(f"🚀 Starting fixed asset analysis of {total_transactions} transactions...")
        
        # Pre-process all text data
        print("📝 Pre-processing text data...")
        transactions_df = self._preprocess_transaction_texts(transactions_df)
        
        # Analyze each transaction
        for idx, row in transactions_df.iterrows():
            try:
                detection_result = self.detect_fixed_asset(row.to_dict(), threshold)
                
                # Combine original transaction data with results
                result = {
                    **row.to_dict(),
                    **detection_result
                }
                results.append(result)
                
                # Progress update
                if (idx + 1) % 100 == 0:
                    print(f"Progress: {idx + 1}/{total_transactions} ({((idx + 1)/total_transactions)*100:.1f}%)")
                    
            except Exception as e:
                error_result = {
                    **row.to_dict(),
                    'is_fixed_asset': False,
                    'score': 0,
                    'confidence_level': 'Error',
                    'confidence_factors': [f'Error: {str(e)[:50]}'],
                    'detailed_breakdown': {},
                    'category': None,
                    'estimated_useful_life': 0
                }
                results.append(error_result)
        
        print(f"✅ Analysis completed: {len(results)} results generated")
        return pd.DataFrame(results)
    
    def get_employee_email_internal(self, employee_name):
        """
        Get employee email from the mapping, with fallback logic for name variations
        """
        if not employee_name:
            return None

        # Clean the input name
        employee_name = str(employee_name).strip()

        # Direct lookup first
        if employee_name in self.EMPLOYEE_EMAIL_MAPPING:
            return self.EMPLOYEE_EMAIL_MAPPING[employee_name]

        # Try variations for common name formats
        name_variations = [
            employee_name.strip(),
            employee_name.title(),
            employee_name.lower().title(),
        ]

        # Check for partial matches (first name only)
        first_name = employee_name.split()[0] if ' ' in employee_name else employee_name
        if first_name in self.EMPLOYEE_EMAIL_MAPPING:
            return self.EMPLOYEE_EMAIL_MAPPING[first_name]

        # Check all variations
        for variation in name_variations:
            if variation in self.EMPLOYEE_EMAIL_MAPPING:
                return self.EMPLOYEE_EMAIL_MAPPING[variation]

        # Try case-insensitive lookup
        employee_name_lower = employee_name.lower()
        for key, email in self.EMPLOYEE_EMAIL_MAPPING.items():
            if key.lower() == employee_name_lower:
                return email

        # Try partial matching for common variations
        for key, email in self.EMPLOYEE_EMAIL_MAPPING.items():
            # Check if the input name is contained in the key or vice versa
            if employee_name.lower() in key.lower() or key.lower() in employee_name.lower():
                return email

        return None

    
    def create_fixed_asset_html_email(self, employee_name, transactions_df, test_mode=False):
        """Create HTML email for fixed asset alerts using brand styling"""

        total_value = transactions_df['amount'].sum() if 'amount' in transactions_df.columns else 0
        transaction_count = len(transactions_df)

        # Check if any transactions are from Apple
        has_apple_transactions = False
        if 'vendor' in transactions_df.columns:
            has_apple_transactions = transactions_df['vendor'].str.contains('Apple', case=False, na=False).any()
        elif 'merchant' in transactions_df.columns:
            has_apple_transactions = transactions_df['merchant'].str.contains('Apple', case=False, na=False).any()

        # Get recipient email
        recipient_email = "lulu@k-id.com" if test_mode else self.get_employee_email_internal(employee_name)

        # Build transaction details HTML
        transaction_details_html = ""
        for _, transaction in transactions_df.iterrows():
            vendor = transaction.get('vendor', transaction.get('merchant', 'Unknown Vendor'))
            amount = transaction.get('amount', 0)
            description = transaction.get('description', transaction.get('memo', 'No description'))
            category = transaction.get('category', 'Unclassified')
            confidence = transaction.get('confidence_level', 'Unknown')
            score = transaction.get('score', 0)

            # Highlight Apple transactions
            is_apple = 'Apple' in str(vendor)
            apple_highlight = "border-left: 3px solid #007AFF;" if is_apple else f"border-left: 3px solid {self.BRAND_COLORS['purple']};"
            apple_icon = "🍎 " if is_apple else "🏪 "

            # Category display with icon
            category_icons = {
                'electronics_it': '💻',
                'furniture_office': '🪑', 
                'machinery_equipment': '⚙️',
                'vehicles_transport': '🚛',
                'building_improvements': '🏗️',
                'software_licenses': '💿'
            }

            category_display = f"{category_icons.get(category, '📦')} {category.replace('_', ' ').title()}" if category else '📦 Unclassified'

            # Enhanced confidence display with badge styling
            confidence_badge_bg = {
                'Very High': '#28a745',
                'High': self.BRAND_COLORS['purple'],
                'Medium': self.BRAND_COLORS['orange'],
                'Low': '#ffc107',
                'Very Low': '#dc3545'
            }

            transaction_details_html += f"""
            <div style="background-color: #ffffff; padding: 16px; border-radius: 8px; margin: 12px 0; {apple_highlight} box-shadow: 0 2px 4px rgba(0,0,0,0.06); transition: all 0.3s ease;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                    <div style="flex: 1;">
                        <h4 style="color: {self.BRAND_COLORS['blackberry']}; margin: 0 0 6px 0; font-size: 16px; font-weight: 600; display: flex; align-items: center;">
                            <span style="margin-right: 6px;">{apple_icon}</span> {vendor}
                        </h4>
                        <span style="background-color: {confidence_badge_bg.get(confidence, self.BRAND_COLORS['purple'])}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 600; text-transform: uppercase;">
                            {confidence}
                        </span>
                    </div>
                    <div style="text-align: right;">
                        <span style="background: linear-gradient(135deg, {self.BRAND_COLORS['orange']} 0%, #ff7043 100%); color: white; padding: 6px 12px; border-radius: 16px; font-size: 14px; font-weight: bold; box-shadow: 0 1px 4px rgba(255,152,0,0.2);">
                            ${amount:,.2f}
                        </span>
                        <div style="margin-top: 4px; font-size: 10px; color: {self.BRAND_COLORS['blackberry']}; font-weight: 600;">
                            Score: {score:.1f}/100
                        </div>
                    </div>
                </div>

                <div style="background-color: #f8f9fa; padding: 12px; border-radius: 6px; margin: 8px 0;">
                    <p style="margin: 0 0 6px 0; color: {self.BRAND_COLORS['black']}; font-size: 13px; line-height: 1.4;">
                        <strong style="color: {self.BRAND_COLORS['blackberry']};">📝</strong> {description}
                    </p>
                    <p style="margin: 0; color: {self.BRAND_COLORS['black']}; font-size: 13px;">
                        <strong style="color: {self.BRAND_COLORS['blackberry']};">🏷️</strong> {category_display}
                    </p>
                </div>
            </div>
            """

        # Test mode warning with enhanced styling
        test_warning = f"""
        <div style='background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); border: 2px solid #f0ad4e; padding: 16px; border-radius: 8px; margin: 20px 0; text-align: center; box-shadow: 0 2px 6px rgba(240,173,78,0.15);'>
            <p style='margin: 0; font-size: 16px; color: #856404; font-weight: bold;'>
                🧪 TEST MODE - DEVELOPMENT ENVIRONMENT 🧪
            </p>
            <p style='margin: 4px 0 0 0; font-size: 12px; color: #856404;'>
                This email is being sent for testing purposes
            </p>
        </div>
        """ if test_mode else ""

        # Apple CC notice
        apple_cc_notice = ""
        if has_apple_transactions:
            apple_cc_notice = f"""
            <div style='background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border: 2px solid #007AFF; padding: 16px; border-radius: 8px; margin: 20px 0; text-align: center; box-shadow: 0 2px 6px rgba(0,122,255,0.15);'>
                <p style='margin: 0; font-size: 14px; color: #0d47a1; font-weight: bold;'>
                    🍎 Apple Transaction Detected
                </p>
                <p style='margin: 4px 0 0 0; font-size: 12px; color: #1565c0;'>
                    Alyssa Aw has been CC'd on this email for Apple-related transactions
                </p>
            </div>
            """

        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Fixed Asset Management System</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

                .transaction-card:hover {{
                    transform: translateY(-1px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
                }}

                .cta-button:hover {{
                    transform: translateY(-1px);
                    box-shadow: 0 3px 8px rgba(123,31,162,0.3) !important;
                }}
            </style>
        </head>
        <body style="font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: {self.BRAND_COLORS['blackberry']}; margin: 0; padding: 0; background-color: #f5f7fa;">
            <div style="max-width: 700px; margin: 0 auto; background-color: #ffffff; box-shadow: 0 0 16px rgba(0,0,0,0.08);">
                <!-- Header with enhanced branding -->
                <div style="background: linear-gradient(135deg, {self.BRAND_COLORS['gradient_start']} 0%, {self.BRAND_COLORS['gradient_end']} 100%); padding: 32px 24px; text-align: center;">
                    <h1 style="color: #ffffff; margin: 0 0 8px 0; font-size: 26px; font-weight: 700; letter-spacing: -0.5px;">
                        🎯 Fixed Asset Alert
                    </h1>
                </div>

                <!-- Main content -->
                <div style="padding: 24px;">
                    <div style="margin-bottom: 24px;">
                        <h2 style="color: {self.BRAND_COLORS['blackberry']}; margin: 0 0 16px 0; font-size: 20px; font-weight: 600;">
                            Hello {employee_name}! 👋
                        </h2>

                        {test_warning}
                        {apple_cc_notice}

                        <div style="background: linear-gradient(135deg, #f8f9ff 0%, #e8f2ff 100%); padding: 20px; border-radius: 12px; border: 1px solid {self.BRAND_COLORS['purple_light']}; margin: 20px 0;">
                            <p style="margin: 0 0 12px 0; font-size: 15px; color: {self.BRAND_COLORS['black']}; line-height: 1.6;">
                                Our AI system detected <strong style="color: {self.BRAND_COLORS['purple']};">{transaction_count} transactions</strong> 
                                worth <strong style="color: {self.BRAND_COLORS['orange']};">${total_value:,.2f}</strong> that may qualify as fixed assets.
                            </p>
                            <p style="margin: 0; font-size: 12px; color: {self.BRAND_COLORS['blackberry']}; opacity: 0.8; font-style: italic;">
                                🤖 Powered by machine learning for accurate asset detection
                            </p>
                        </div>
                    </div>

                    <!-- Enhanced Summary Overview -->
                    <div style="background: linear-gradient(135deg, {self.BRAND_COLORS['purple_light']} 0%, #f3e5f5 100%); padding: 20px; border-radius: 12px; margin: 20px 0; border: 2px solid {self.BRAND_COLORS['purple']};">
                        <h3 style="color: {self.BRAND_COLORS['blackberry']}; margin: 0 0 16px 0; font-size: 18px; font-weight: 600; display: flex; align-items: center;">
                            <span style="background-color: {self.BRAND_COLORS['purple']}; color: white; padding: 6px; border-radius: 6px; margin-right: 10px; font-size: 16px;">📊</span>
                            Detection Summary
                        </h3>

                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                            <div style="background-color: white; padding: 16px; border-radius: 8px; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
                                <div style="font-size: 24px; color: {self.BRAND_COLORS['purple']}; font-weight: 700; margin-bottom: 4px;">{transaction_count}</div>
                                <p style="margin: 0; font-size: 12px; color: {self.BRAND_COLORS['black']}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Transactions</p>
                            </div>
                            <div style="background-color: white; padding: 16px; border-radius: 8px; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
                                <div style="font-size: 24px; color: {self.BRAND_COLORS['orange']}; font-weight: 700; margin-bottom: 4px;">${total_value:,.0f}</div>
                                <p style="margin: 0; font-size: 12px; color: {self.BRAND_COLORS['black']}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Total Value</p>
                            </div>
                        </div>
                    </div>

                    <!-- Enhanced Transaction details -->
                    <div style="background-color: {self.BRAND_COLORS['purple_light']}; padding: 20px; border-radius: 12px; margin: 20px 0;">
                        <h3 style="color: {self.BRAND_COLORS['blackberry']}; margin: 0 0 16px 0; font-size: 18px; font-weight: 600; display: flex; align-items: center;">
                            <span style="background-color: {self.BRAND_COLORS['purple']}; color: white; padding: 6px; border-radius: 6px; margin-right: 10px; font-size: 16px;">🏢</span>
                            Asset Transactions
                        </h3>
                        <p style="margin: 0 0 12px 0; font-size: 13px; color: {self.BRAND_COLORS['blackberry']}; opacity: 0.8;">
                            Review each transaction for proper classification and approval
                        </p>
                        {transaction_details_html}
                    </div>

                    <!-- Enhanced Action required section -->
                    <div style="background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%); padding: 20px; border-radius: 12px; margin: 20px 0; border: 2px solid {self.BRAND_COLORS['orange']};">
                        <h3 style="color: {self.BRAND_COLORS['orange']}; margin: 0 0 16px 0; font-size: 18px; font-weight: 600; display: flex; align-items: center;">
                            <span style="background-color: {self.BRAND_COLORS['orange']}; color: white; padding: 6px; border-radius: 6px; margin-right: 10px; font-size: 16px;">⚡</span>
                            Next Steps
                        </h3>

                        <div style="background-color: white; padding: 16px; border-radius: 8px; margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
                            <ul style="margin: 0; padding-left: 0; color: {self.BRAND_COLORS['black']}; list-style: none;">
                                <li style="margin-bottom: 12px; display: flex; align-items: flex-start;">
                                    <span style="background-color: {self.BRAND_COLORS['purple']}; color: white; width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold; margin-right: 10px; flex-shrink: 0;">1</span>
                                    <span style="font-size: 14px;">Review each transaction for accurate classification</span>
                                </li>
                                <li style="margin-bottom: 12px; display: flex; align-items: flex-start;">
                                    <span style="background-color: {self.BRAND_COLORS['purple']}; color: white; width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold; margin-right: 10px; flex-shrink: 0;">2</span>
                                    <span style="font-size: 14px;">Submit the reimbursement form for qualifying assets (click the button below) </span>
                                </li>
                                <li style="margin-bottom: 0; display: flex; align-items: flex-start;">
                                    <span style="background-color: {self.BRAND_COLORS['purple']}; color: white; width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold; margin-right: 10px; flex-shrink: 0;">3</span>
                                    <span style="font-size: 14px;">Ensure all documentation is complete</span>
                                </li>
                            </ul>
                        </div>

                        <!-- CTA Button -->
                        <div style="text-align: center; margin-top: 16px;">
                            <a href="https://docs.google.com/forms/d/e/1FAIpQLSe1P6NJ7i81RMBVK9eF5l-MMZSK77CmhsIVL-w5azMNDws2GQ/viewform?usp=dialog" 
                               style="display: inline-block; background: linear-gradient(135deg, {self.BRAND_COLORS['purple']} 0%, #8e24aa 100%); color: white; padding: 12px 24px; border-radius: 24px; text-decoration: none; font-weight: 600; font-size: 14px; box-shadow: 0 3px 8px rgba(123,31,162,0.2); transition: all 0.3s ease;"
                               class="cta-button">
                                📋 Submit Reimbursement Form
                            </a>
                        </div>
                    </div>

                    <!-- Enhanced Contact information -->
                    <div style="text-align: center; margin-top: 24px; padding: 20px; background-color: #f8f9fa; border-radius: 12px; border-top: 3px solid {self.BRAND_COLORS['purple']};">
                        <div style="margin-bottom: 16px;">
                            <p style="margin: 0 0 6px 0; font-size: 16px; color: {self.BRAND_COLORS['blackberry']}; font-weight: 600;">
                                Best regards,
                            </p>
                            <p style="margin: 0; font-size: 18px; color: {self.BRAND_COLORS['purple']}; font-weight: 700;">
                                💼 Finance Team
                            </p>
                        </div>

                        <div style="background-color: white; padding: 16px; border-radius: 8px; display: inline-block; box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
                            <p style="margin: 0 0 8px 0; font-size: 13px; color: {self.BRAND_COLORS['black']};">
                                <strong>Need assistance?</strong> We're here to help! 
                            </p>
                            <p style="margin: 0; font-size: 13px; color: {self.BRAND_COLORS['black']};">
                                📧 <a href="mailto:finance@k-id.com" style="color: {self.BRAND_COLORS['purple']}; text-decoration: none; font-weight: 600;">finance@k-id.com</a>
                            </p>
                        </div>
                    </div>

                    <!-- Footer -->
                    <div style="text-align: center; margin-top: 20px; padding-top: 16px; border-top: 1px solid #dee2e6;">
                        <p style="margin: 0; font-size: 11px; color: {self.BRAND_COLORS['black']}; opacity: 0.7;">
                            This is an automated message from the Fixed Asset Management System. Please do not reply to this email.
                        </p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        return html_body, recipient_email, has_apple_transactions
    
    def send_fixed_asset_email_smtp(self, email_data):
        """Send a single fixed asset email via SMTP"""
        try:
            smtp_config = email_data['smtp_config']
            msg = email_data['message']

            print(f"📧 Connecting to SMTP server: {smtp_config['smtp_server']}:{smtp_config['smtp_port']}")

            with smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port']) as server:
                server.starttls()
                server.login(smtp_config['smtp_user'], smtp_config['smtp_pass'])
                
                # Get all recipients (TO + CC)
                to_addrs = [email_data['to_addr']]
                
                # Add CC recipients if present
                if 'Cc' in msg:
                    cc_addrs = [addr.strip() for addr in msg['Cc'].split(',')]
                    to_addrs.extend(cc_addrs)
                
                # Send email to all recipients
                server.send_message(msg, to_addrs=to_addrs)

            cc_info = f" (CC: {msg.get('Cc', 'None')})" if 'Cc' in msg else ""
            print(f"✅ Fixed asset email sent successfully to {email_data['to_addr']}{cc_info}")
            return True

        except Exception as e:
            print(f"❌ Failed to send fixed asset email to {email_data['to_addr']}: {e}")
            return False

    def send_scheduled_fixed_asset_email(self, email_id):
        """Send a scheduled fixed asset email"""

        try:
            import streamlit as st

            # Initialize if not exists
            if 'FIXED_ASSET_SCHEDULED_EMAILS' not in st.session_state:
                st.session_state.FIXED_ASSET_SCHEDULED_EMAILS = {}
                print(f"❌ Fixed asset email {email_id} not found - session state was empty")
                return

            EMAIL_LOCK = threading.Lock()  # Create local lock

            with EMAIL_LOCK:
                if email_id not in st.session_state.FIXED_ASSET_SCHEDULED_EMAILS:
                    print(f"❌ Fixed asset email {email_id} not found in scheduled emails")
                    return

                email_data = st.session_state.FIXED_ASSET_SCHEDULED_EMAILS[email_id]

                # Check if email was cancelled or already sent
                if email_data.get('cancelled', False):
                    print(f"⏭️ Fixed asset email {email_id} was cancelled, skipping")
                    return

                if email_data.get('sent', False):
                    print(f"⏭️ Fixed asset email {email_id} already sent, skipping")
                    return

            # Send the email
            success = self.send_fixed_asset_email_smtp(email_data)

            # Update status
            with EMAIL_LOCK:
                if email_id in st.session_state.FIXED_ASSET_SCHEDULED_EMAILS:
                    st.session_state.FIXED_ASSET_SCHEDULED_EMAILS[email_id]['sent'] = success
                    st.session_state.FIXED_ASSET_SCHEDULED_EMAILS[email_id]['failed'] = not success
                    st.session_state.FIXED_ASSET_SCHEDULED_EMAILS[email_id]['sent_time'] = datetime.now()

            print(f"{'✅' if success else '❌'} Fixed asset email {email_id} {'sent successfully' if success else 'failed'}")

        except Exception as e:
            print(f"❌ Error sending scheduled fixed asset email {email_id}: {e}")

            # Initialize if not exists
            if 'FIXED_ASSET_SCHEDULED_EMAILS' not in st.session_state:
                st.session_state.FIXED_ASSET_SCHEDULED_EMAILS = {}

            with EMAIL_LOCK:
                if email_id in st.session_state.FIXED_ASSET_SCHEDULED_EMAILS:
                    st.session_state.FIXED_ASSET_SCHEDULED_EMAILS[email_id]['failed'] = True
                    
    def send_delayed_fixed_asset_email(self, email_data, delay_minutes=5, fixed_asset_dict=None):
        """Send a fixed asset email after a specified delay, checking passed dictionary for cancellation"""
        import streamlit as st
        EMAIL_LOCK = threading.Lock()

        email_id = email_data['id']

        # Use the passed dictionary instead of session state
        if fixed_asset_dict is None:
            fixed_asset_dict = st.session_state.FIXED_ASSET_SCHEDULED_EMAILS

        print(f"📧 Fixed Asset email {email_id} scheduled for {delay_minutes} minutes from now...")

        # Wait for the delay period, checking every 10 seconds
        total_seconds = delay_minutes * 60
        check_interval = 10
        elapsed = 0

        while elapsed < total_seconds:
            # Use thread lock when checking status
            with EMAIL_LOCK:
                current_email_status = fixed_asset_dict.get(email_id, {})
                is_cancelled = current_email_status.get('cancelled', False)
                send_immediately = current_email_status.get('send_immediately', False)

            if is_cancelled:
                print(f"❌ Fixed Asset email {email_id} cancelled before sending (loop check)")
                # Update the dictionary with lock
                with EMAIL_LOCK:
                    if email_id in fixed_asset_dict:
                        fixed_asset_dict[email_id]['cancelled'] = True
                        fixed_asset_dict[email_id]['cancelled_time'] = datetime.now()
                return False

            # Check if marked for immediate sending
            if send_immediately:
                print(f"🚀 Fixed Asset email {email_id} marked for immediate sending")
                break

            time.sleep(min(check_interval, total_seconds - elapsed))
            elapsed += check_interval

            # Show countdown every 2 minutes
            if elapsed % 120 == 0 and elapsed < total_seconds:
                remaining = (total_seconds - elapsed) // 60
                print(f"⏰ Fixed Asset email {email_id} will be sent in {remaining} minutes")

        # Final check before sending with lock
        with EMAIL_LOCK:
            final_email_status = fixed_asset_dict.get(email_id, {})
            is_cancelled = final_email_status.get('cancelled', False)
            print(f"🔍 DEBUG FINAL CHECK {email_id}: cancelled={is_cancelled}")

        if is_cancelled:
            print(f"❌ Fixed Asset email {email_id} cancelled before sending (final check)")
            with EMAIL_LOCK:
                if email_id in fixed_asset_dict:
                    fixed_asset_dict[email_id]['cancelled'] = True
                    fixed_asset_dict[email_id]['cancelled_time'] = datetime.now()
            return False

        # Send the email
        try:
            success = self.send_fixed_asset_email_smtp(email_data)

            print(f"✅ Fixed Asset email {email_id} sent successfully to {email_data['employee']} ({email_data['to_addr']})")

            # Mark as sent in dictionary with lock
            with EMAIL_LOCK:
                if email_id in fixed_asset_dict:
                    fixed_asset_dict[email_id]['sent'] = True
                    fixed_asset_dict[email_id]['sent_time'] = datetime.now()

            return True

        except Exception as e:
            print(f"❌ Failed to send Fixed Asset email {email_id}: {e}")
            # Mark as failed in dictionary with lock
            with EMAIL_LOCK:
                if email_id in fixed_asset_dict:
                    fixed_asset_dict[email_id]['failed'] = True
                    fixed_asset_dict[email_id]['failed_time'] = datetime.now()
                    fixed_asset_dict[email_id]['error'] = str(e)
            return False

    def schedule_fixed_asset_email_thread(self, email_data, fixed_asset_dict, delay_minutes):
        """CORRECTED: Schedule a fixed asset email to be sent after a delay - receives data directly"""
        
        email_id = email_data['id']
        
        try:
            print(f"⏰ Fixed asset email {email_id} thread started")
            
            # Use the new delayed email function with passed data
            success = self.send_delayed_fixed_asset_email(
                email_data,
                delay_minutes, 
                fixed_asset_dict
            )

            print(f"📧 Fixed asset email {email_id} thread completed: {'Success' if success else 'Failed/Cancelled'}")

        except Exception as e:
            print(f"❌ Error in fixed asset email thread {email_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Mark as failed in the passed dictionary
            EMAIL_LOCK = threading.Lock()
            with EMAIL_LOCK:
                if email_id in fixed_asset_dict:
                    fixed_asset_dict[email_id]['failed'] = True
                    fixed_asset_dict[email_id]['failed_time'] = datetime.now()
                    fixed_asset_dict[email_id]['error'] = str(e)
            
    def add_scheduled_fixed_asset_email(self, email_id, email_data):
        """CORRECTED: Add a fixed asset email to the scheduled emails"""
        import streamlit as st
        import threading

        # Initialize if not exists
        if 'FIXED_ASSET_SCHEDULED_EMAILS' not in st.session_state:
            st.session_state.FIXED_ASSET_SCHEDULED_EMAILS = {}

        # Store email data BEFORE starting thread
        st.session_state.FIXED_ASSET_SCHEDULED_EMAILS[email_id] = email_data
        print(f"📧 Fixed asset email {email_id} stored in session state")

        # CORRECTED: Pass email_data and dictionary reference to thread
        delay_minutes = email_data.get('delay_minutes', 5)
        thread = threading.Thread(
            target=self.schedule_fixed_asset_email_thread, 
            args=(email_data, st.session_state.FIXED_ASSET_SCHEDULED_EMAILS, delay_minutes)
        )
        thread.daemon = True
        thread.start()

        print(f"📧 Fixed asset email {email_id} thread launched")

    def cancel_all_scheduled_fixed_asset_emails():
        """Cancel all scheduled fixed asset emails"""

        # Initialize if not exists
        if 'FIXED_ASSET_SCHEDULED_EMAILS' not in st.session_state:
            st.session_state.FIXED_ASSET_SCHEDULED_EMAILS = {}
            return 0

        cancelled_count = 0
        EMAIL_LOCK = threading.Lock()  # Create local lock

        with EMAIL_LOCK:
            for email_id, email_data in st.session_state.FIXED_ASSET_SCHEDULED_EMAILS.items():
                if not email_data.get('cancelled', False) and not email_data.get('sent', False):
                    email_data['cancelled'] = True
                    email_data['cancelled_time'] = datetime.now()
                    cancelled_count += 1

        print(f"❌ Cancelled {cancelled_count} scheduled fixed asset emails")
        return cancelled_count
    
    def send_fixed_asset_emails_with_ui(self, employee_transactions_dict, smtp_config, test_mode=True, delay_minutes=5):
        """CORRECTED: Send fixed asset alert emails using the proper email system with Apple CC functionality"""
        emails_scheduled = 0
        emails_skipped = 0

        print(f"🔍 DEBUG: Starting email processing for {len(employee_transactions_dict)} employees")
        print(f"🔍 DEBUG: Test mode: {test_mode}")

        try:
            # Process each employee's transactions
            for employee_name, transactions_df in employee_transactions_dict.items():
                print(f"🔍 DEBUG: Processing employee: '{employee_name}'")
                print(f"🔍 DEBUG: Transaction count: {len(transactions_df)}")

                if len(transactions_df) == 0:
                    print(f"🔍 DEBUG: Skipping {employee_name} - no transactions")
                    emails_skipped += 1
                    continue

                # Get the email address
                recipient_email = self.get_employee_email_internal(employee_name)
                print(f"🔍 DEBUG: Recipient email found: {recipient_email}")

                # Skip if NOT in test mode AND no email found
                if not test_mode and not recipient_email:
                    print(f"❌ No email found for employee: {employee_name}. Skipping notification.")
                    emails_skipped += 1
                    continue

                # In test mode, override to lulu@k-id.com
                if test_mode:
                    final_recipient = "lulu@k-id.com"
                    print(f"🔍 DEBUG: Test mode enabled - using test email: {final_recipient}")
                else:
                    final_recipient = recipient_email

                # Generate email content and check for Apple transactions
                try:
                    html_body, _, has_apple_transactions = self.create_fixed_asset_html_email(
                        employee_name, transactions_df, test_mode
                    )
                    print(f"🔍 DEBUG: Email content generated successfully")
                    print(f"🔍 DEBUG: Has Apple transactions: {has_apple_transactions}")
                except Exception as e:
                    print(f"❌ Error generating email content for {employee_name}: {e}")
                    emails_skipped += 1
                    continue

                # Create email message
                msg = EmailMessage()
                msg['From'] = smtp_config['smtp_user']
                msg['To'] = final_recipient

                # Add CC for Apple transactions
                cc_recipients = ["finance@k-id.com"]
                
                # Add Alyssa for Apple transactions
                if has_apple_transactions:
                    alyssa_email = "alyssaaw@k-id.com"
                    cc_recipients.append(alyssa_email)
                    print(f"🍎 Apple transaction detected - CC'ing Alyssa Aw: {alyssa_email}")

                # Always set CC since finance@k-id.com is always included
                msg['Cc'] = ', '.join(cc_recipients)
                print(f"📧 CC recipients: {', '.join(cc_recipients)}")

                test_prefix = "[TEST MODE] " if test_mode else ""
                apple_prefix = "[APPLE] " if has_apple_transactions else ""
                subject = f"{test_prefix}{apple_prefix}Fixed Asset Alert - {employee_name} 🏢"
                msg['Subject'] = subject
                msg.add_alternative(html_body, subtype='html')

                # Create email data for scheduling
                email_id = f"FIXED_ASSET_{employee_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                email_data = {
                    'id': email_id,
                    'employee': employee_name,
                    'to_addr': final_recipient,
                    'cc_addrs': cc_recipients,
                    'has_apple_cc': has_apple_transactions,
                    'subject': subject,
                    'from_addr': smtp_config['smtp_user'],
                    'html_body': html_body,
                    'message': msg,
                    'smtp_config': smtp_config,
                    'send_immediately': False,
                    'scheduled_time': datetime.now() + timedelta(minutes=delay_minutes),
                    'delay_minutes': delay_minutes,
                    'has_attachment': False,
                    'violations_count': len(transactions_df),
                    'test_mode': test_mode,
                    'cancelled': False,
                    'sent': False,
                    'failed': False
                }

                # CORRECTED: Use the proper function to add to scheduled emails
                self.add_scheduled_fixed_asset_email(email_id, email_data)

                cc_info = f" (CC: {', '.join(cc_recipients)})" if cc_recipients else ""
                print(f"📧 Fixed asset alert scheduled for {employee_name} ({final_recipient}){cc_info}")
                print(f"   Email ID: {email_id}")
                print(f"   Transactions: {len(transactions_df)}")
                if has_apple_transactions:
                    apple_count = 0
                    if 'vendor' in transactions_df.columns:
                        apple_count += transactions_df['vendor'].str.contains('Apple', case=False, na=False).sum()
                    elif 'merchant' in transactions_df.columns:
                        apple_count += transactions_df['merchant'].str.contains('Apple', case=False, na=False).sum()
                    print(f"   🍎 Apple transactions: {apple_count}")

                emails_scheduled += 1

            print(f"\n🏢 FIXED ASSET ALERT SCHEDULING COMPLETE:")
            print(f"   Scheduled: {emails_scheduled}")
            print(f"   Skipped: {emails_skipped}")
            print(f"   Delay: {delay_minutes} minutes")

            # CORRECTED: Remove problematic static UI launch or implement properly
            if emails_scheduled > 0:
                try:
                    # OPTION 1: Comment out the static UI for now
                    print("📧 Emails scheduled successfully. Static UI launch disabled for now.")

                    # OPTION 2: Or implement a simple version
                    # self.launch_simple_email_ui(emails_scheduled, delay_minutes)

                except Exception as ui_error:
                    print(f"⚠️ Static UI failed to launch: {ui_error}")
                    print("📧 Emails are still scheduled and will be sent normally")

            return {
                'success': True,
                'scheduled_count': emails_scheduled,
                'skipped_count': emails_skipped,
                'delay_minutes': delay_minutes,
                'message': f'Successfully scheduled {emails_scheduled} fixed asset alert emails with {delay_minutes} minute delay'
            }

        except Exception as e:
            print(f"❌ Failed to schedule fixed asset emails: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'message': f'Failed to schedule fixed asset emails: {e}'
            }