from collections import defaultdict

class ItemData:
    def __init__(self, item_id, pool_data):
        # Unique identifier and original pool data
        self.item_id = item_id
        self.pool_data = pool_data

        # Placeholder for associated metadata (should be a dict)
        self.metadata = None  

        # List of review dicts for this item
        self.reviews = []     

        # Container for tags/labels (added by the LLM algorithm)
        self.tags = []        

    def add_metadata(self, metadata):
        # Attach metadata (assuming one metadata entry per item)
        self.metadata = metadata

    def add_review(self, review):
        # Append a new review to the reviews list
        self.reviews.append(review)

    def remove_origin_reviews(self, origin_reviews):
        """
        Remove reviews that match the origin (e.g. ori_review in queries).
        `origin_reviews` could be a set of review texts or ids that need to be filtered out.
        """
        self.reviews = [r for r in self.reviews if r.get('text') not in origin_reviews]

    def add_tag(self, tag):
        # Attach a new tag/label discovered during processing
        self.tags.append(tag)

    def __repr__(self):
        # For convenience during debugging/printing
        return (f"ItemData(item_id={self.item_id}, "
                f"metadata_set={'Yes' if self.metadata else 'No'}, "
                f"num_reviews={len(self.reviews)}, "
                f"tags={self.tags})")
