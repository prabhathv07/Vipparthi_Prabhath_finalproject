#!/usr/bin/env python
# coding: utf-8

# In[17]:


"""
CS634 Midterm Project: Frequent Itemset Mining
Complete Implementation with Brute Force, Apriori, and FP-Growth Algorithms

Name: [Prabhath Vinay Vipparthi]
"""

import pandas as pd
import itertools
import os
import time
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder


class DataManager:
    """Handles dataset loading and verification"""
    
    @staticmethod
    def get_available_datasets():
        """Get list of available datasets in current directory"""
        datasets = []
        required_datasets = ['amazon', 'bestbuy', 'walmart', 'target', 'kroger']
        
        for dataset in required_datasets:
            if os.path.exists(f"{dataset}.csv"):
                datasets.append(dataset)
        
        return sorted(datasets)
    
    @staticmethod
    def load_dataset(dataset_name):
        """Load dataset from CSV file"""
        filename = f"{dataset_name}.csv"
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Dataset {filename} not found")
        
        df = pd.read_csv(filename)
        transactions = []
        
        for _, row in df.iterrows():
            items = row['items'].split(',')
            transactions.append(items)
        
        return transactions


class BruteForceAlgorithm:
    """Implements brute force frequent itemset mining"""
    
    def __init__(self, transactions):
        self.transactions = transactions
        self.num_transactions = len(transactions)
        self.frequent_itemsets = {}
        self.association_rules = []
    
    def get_unique_items(self):
        """Get all unique items from transactions"""
        unique_items = set()
        for transaction in self.transactions:
            unique_items.update(transaction)
        return sorted(list(unique_items))
    
    def calculate_support(self, itemset):
        """Calculate support for a given itemset"""
        count = 0
        for transaction in self.transactions:
            if set(itemset).issubset(set(transaction)):
                count += 1
        return count / self.num_transactions
    
    def find_frequent_itemsets(self, min_support):
        """Find all frequent itemsets using brute force approach"""
        unique_items = self.get_unique_items()
        self.frequent_itemsets = {}
        
        k = 1
        while True:
            k_itemsets = list(itertools.combinations(unique_items, k))
            
            frequent_k_itemsets = []
            for itemset in k_itemsets:
                support = self.calculate_support(itemset)
                if support >= min_support:
                    frequent_k_itemsets.append({
                        'itemset': itemset,
                        'support': support
                    })
            
            if not frequent_k_itemsets:
                break
            
            self.frequent_itemsets[k] = frequent_k_itemsets
            k += 1
            if k > len(unique_items):
                break
        
        return self.frequent_itemsets
    
    def generate_association_rules(self, min_confidence):
        """Generate association rules from frequent itemsets"""
        self.association_rules = []
        
        for k in range(2, len(self.frequent_itemsets) + 1):
            for itemset_info in self.frequent_itemsets[k]:
                itemset = itemset_info['itemset']
                itemset_support = itemset_info['support']
                
                for i in range(1, len(itemset)):
                    for antecedent in itertools.combinations(itemset, i):
                        antecedent = set(antecedent)
                        consequent = set(itemset) - antecedent
                        
                        antecedent_support = self.calculate_support(tuple(antecedent))
                        if antecedent_support > 0:
                            confidence = itemset_support / antecedent_support
                            
                            if confidence >= min_confidence:
                                rule = {
                                    'antecedent': tuple(antecedent),
                                    'consequent': tuple(consequent),
                                    'support': itemset_support,
                                    'confidence': confidence
                                }
                                self.association_rules.append(rule)
        
        return self.association_rules


class LibraryAlgorithms:
    """Handles Apriori and FP-Growth using mlxtend library"""
    
    @staticmethod
    def run_apriori(transactions, min_support, min_confidence):
        """Run Apriori algorithm with error handling"""
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        encoded_df = pd.DataFrame(te_array, columns=te.columns_)
        
        frequent_itemsets = apriori(encoded_df, min_support=min_support, use_colnames=True)
        
        # Handle case when no frequent itemsets found
        if len(frequent_itemsets) == 0:
            return frequent_itemsets, pd.DataFrame()
        
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        return frequent_itemsets, rules
    
    @staticmethod
    def run_fpgrowth(transactions, min_support, min_confidence):
        """Run FP-Growth algorithm with error handling"""
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        encoded_df = pd.DataFrame(te_array, columns=te.columns_)
        
        frequent_itemsets = fpgrowth(encoded_df, min_support=min_support, use_colnames=True)
        
        # Handle case when no frequent itemsets found
        if len(frequent_itemsets) == 0:
            return frequent_itemsets, pd.DataFrame()
        
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        return frequent_itemsets, rules


class InputValidator:
    """Handles user input validation with defensive programming"""
    
    @staticmethod
    def validate_float_input(prompt, min_val=0.0, max_val=1.0):
        """Validate float input"""
        while True:
            try:
                value = float(input(prompt))
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"Please enter a value between {min_val} and {max_val}")
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
    
    @staticmethod
    def validate_int_input(prompt, min_val, max_val):
        """Validate integer input"""
        while True:
            try:
                value = int(input(prompt))
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"Please enter a number between {min_val} and {max_val}")
            except ValueError:
                print("Invalid input. Please enter a valid number.")


class ResultDisplayer:
    """Handles display of results in professional format"""
    
    @staticmethod
    def display_dataset_info(datasets):
        """Display available datasets"""
        print("AVAILABLE DATABASES:")
        for i, dataset in enumerate(datasets, 1):
            print(f"  {i}. {dataset}")
    
    @staticmethod
    def display_algorithm_results(algorithm_name, execution_time, itemsets_count, rules_count, 
                                sample_itemsets=None, sample_rules=None):
        """Display results for a specific algorithm"""
        print(f"\n{algorithm_name} RESULTS:")
        print("-" * 50)
        print(f"Execution Time: {execution_time:.4f} seconds")
        print(f"Frequent Itemsets Found: {itemsets_count}")
        print(f"Association Rules Generated: {rules_count}")
        
        if itemsets_count == 0:
            print("No frequent itemsets found with current support threshold.")
            print("Try lowering the minimum support value.")
            return
        
        if sample_itemsets and len(sample_itemsets) > 0:
            print(f"\nSample Frequent Itemsets:")
            for i, itemset_info in enumerate(sample_itemsets[:3]):
                if isinstance(itemset_info, dict) and 'itemset' in itemset_info:
                    itemset_str = ", ".join(itemset_info['itemset'])
                    support = itemset_info['support']
                else:
                    itemset_str = ", ".join(list(itemset_info['itemsets']))
                    support = itemset_info['support']
                print(f"  {i+1}. {{{itemset_str}}} - Support: {support:.3f}")
        
        if rules_count > 0 and sample_rules and len(sample_rules) > 0:
            print(f"\nSample Association Rules:")
            for i, rule in enumerate(sample_rules[:3]):
                if 'antecedent' in rule:  # Brute force format
                    ant_str = ", ".join(rule['antecedent'])
                    cons_str = ", ".join(rule['consequent'])
                    support = rule['support']
                    confidence = rule['confidence']
                else:  # Library format
                    ant_str = ", ".join(list(rule['antecedents']))
                    cons_str = ", ".join(list(rule['consequents']))
                    support = rule['support']
                    confidence = rule['confidence']
                
                print(f"  {i+1}. {{{ant_str}}} => {{{cons_str}}}")
                print(f"      Support: {support:.3f}, Confidence: {confidence:.3f}")
        elif rules_count == 0 and itemsets_count > 0:
            print("No association rules found with current confidence threshold.")
            print("Try lowering the minimum confidence value.")
    
    @staticmethod
    def display_comparison_summary(results):
        """Display final comparison summary"""
        print("\n" + "=" * 60)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("=" * 60)
        
        print(f"{'Algorithm':<15} {'Time (s)':<12} {'Itemsets':<12} {'Rules':<12}")
        print("-" * 60)
        
        for algo_name, algo_results in results.items():
            print(f"{algo_name:<15} {algo_results['time']:<12.4f} "
                  f"{algo_results['itemsets']:<12} {algo_results['rules']:<12}")


def main():
    """Main function - Complete project implementation"""
    print("CS634 MIDTERM PROJECT: FREQUENT ITEMSET MINING")
    print("=" * 60)
    
    # Part 1: Data Verification
    print("PART 1: DATASET VERIFICATION")
    print("-" * 40)
    
    datasets = DataManager.get_available_datasets()
    if len(datasets) != 5:
        print("Error: Not all 5 datasets found. Please ensure these files exist:")
        print("amazon.csv, bestbuy.csv, walmart.csv, target.csv, kroger.csv")
        return
    
    ResultDisplayer.display_dataset_info(datasets)
    print("All 5 datasets verified and available")
    
    # Part 4: User Input
    print("\nPART 4: USER INPUT PARAMETERS")
    print("-" * 40)
    
    dataset_choice = InputValidator.validate_int_input(
        f"Select a database (1-{len(datasets)}): ", 1, len(datasets)
    )
    selected_dataset = datasets[dataset_choice - 1]
    
    min_support = InputValidator.validate_float_input("Enter minimum support (0.0-1.0): ")
    min_confidence = InputValidator.validate_float_input("Enter minimum confidence (0.0-1.0): ")
    
    # Load selected dataset
    print(f"\nLoading dataset: {selected_dataset}")
    transactions = DataManager.load_dataset(selected_dataset)
    print(f"Loaded {len(transactions)} transactions")
    
    print(f"\nANALYSIS PARAMETERS:")
    print(f"  Dataset: {selected_dataset}")
    print(f"  Minimum Support: {min_support}")
    print(f"  Minimum Confidence: {min_confidence}")
    print("\n" + "=" * 60)
    
    # Part 2: Brute Force Algorithm
    print("\nPART 2: BRUTE FORCE ALGORITHM")
    print("-" * 40)
    
    start_time = time.time()
    bf = BruteForceAlgorithm(transactions)
    bf_itemsets = bf.find_frequent_itemsets(min_support)
    bf_rules = bf.generate_association_rules(min_confidence)
    bf_time = time.time() - start_time
    
    total_bf_itemsets = sum(len(itemsets) for itemsets in bf_itemsets.values())
    
    # Prepare samples for display
    bf_sample_itemsets = []
    for k in bf_itemsets:
        bf_sample_itemsets.extend(bf_itemsets[k])
    
    ResultDisplayer.display_algorithm_results(
        "BRUTE FORCE", bf_time, total_bf_itemsets, len(bf_rules),
        bf_sample_itemsets, bf_rules
    )
    
    # Part 3: Library Algorithms
    print("\nPART 3: APRIORI & FP-GROWTH ALGORITHMS")
    print("-" * 40)
    
    # Apriori
    print("\nRunning Apriori Algorithm...")
    start_time = time.time()
    try:
        apriori_itemsets, apriori_rules = LibraryAlgorithms.run_apriori(
            transactions, min_support, min_confidence
        )
        apriori_time = time.time() - start_time
        
        # Convert to lists for display
        apriori_itemsets_list = apriori_itemsets.to_dict('records') if len(apriori_itemsets) > 0 else []
        apriori_rules_list = apriori_rules.to_dict('records') if len(apriori_rules) > 0 else []
        
        ResultDisplayer.display_algorithm_results(
            "APRIORI", apriori_time, len(apriori_itemsets), len(apriori_rules),
            apriori_itemsets_list, apriori_rules_list
        )
    except Exception as e:
        print(f"Apriori Algorithm Error: {e}")
        apriori_time = 0
        apriori_itemsets = pd.DataFrame()
        apriori_rules = pd.DataFrame()
    
    # FP-Growth
    print("\nRunning FP-Growth Algorithm...")
    start_time = time.time()
    try:
        fpgrowth_itemsets, fpgrowth_rules = LibraryAlgorithms.run_fpgrowth(
            transactions, min_support, min_confidence
        )
        fpgrowth_time = time.time() - start_time
        
        # Convert to lists for display
        fpgrowth_itemsets_list = fpgrowth_itemsets.to_dict('records') if len(fpgrowth_itemsets) > 0 else []
        fpgrowth_rules_list = fpgrowth_rules.to_dict('records') if len(fpgrowth_rules) > 0 else []
        
        ResultDisplayer.display_algorithm_results(
            "FP-GROWTH", fpgrowth_time, len(fpgrowth_itemsets), len(fpgrowth_rules),
            fpgrowth_itemsets_list, fpgrowth_rules_list
        )
    except Exception as e:
        print(f"FP-Growth Algorithm Error: {e}")
        fpgrowth_time = 0
        fpgrowth_itemsets = pd.DataFrame()
        fpgrowth_rules = pd.DataFrame()
    
    # Final Summary
    comparison_results = {
        'Brute Force': {'time': bf_time, 'itemsets': total_bf_itemsets, 'rules': len(bf_rules)},
        'Apriori': {'time': apriori_time, 'itemsets': len(apriori_itemsets), 'rules': len(apriori_rules)},
        'FP-Growth': {'time': fpgrowth_time, 'itemsets': len(fpgrowth_itemsets), 'rules': len(fpgrowth_rules)}
    }
    
    ResultDisplayer.display_comparison_summary(comparison_results)
    
    print(f"\nPROJECT COMPLETED SUCCESSFULLY!")
    print(f"All parts implemented: Data Verification, Brute Force, Apriori, FP-Growth, User Interface")


if __name__ == "__main__":
    main()



# In[ ]:




