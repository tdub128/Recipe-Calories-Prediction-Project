# Interpreting Categorical Features on Recipes to Study and Make Predictions on Their Nutritional Components
## Introduction
### Research Question
#### This research investigates the accuracy of recipe information, focusing on textual features such as descriptions, ingredients, tags, names, as well as recipe ratings to assess their alignment with nutritional components. By analyzing these features, I aim to predict nutritional values without relying on the provided data. While classifier created in this report focuses on predicting sugar level, it provides helpful insights for development of the ultimate model that predicts multiple nutritional values (e.g., calories, fat, protein). This study not only evaluates the reliability of recipe claims but also explores patterns linking text features to nutrition, with implications for improving online recipe data and dietary recommendations.

### Dataset Information

#### The first dataset, `recipe` has a shape of (83,782,10). It contains 83,782 recipes, and 10 attributes of each recipe as described in the following:

| Column          | Description                                                                                       |
|------------------|---------------------------------------------------------------------------------------------------|
| `name`          | Recipe name                                                                                      |
| `id`            | Recipe ID                                                                                        |
| `minutes`       | Minutes to prepare recipe                                                                        |
| `contributor_id`| User ID who submitted this recipe                                                                |
| `submitted`     | Date recipe was submitted                                                                        |
| `tags`          | Food.com tags for recipe                                                                         |
| `nutrition`     | Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for “percentage of daily value” |
| `n_steps`       | Number of steps in recipe                                                                        |
| `steps`         | Text for recipe steps, in order                                                                  |
| `description`   | User-provided description                                                                        |
| `ingredients`   | Text for recipe ingredients                                                                      |
| `n_ingredients` | Number of ingredients in recipe                                                                  |

---

#### The second dataset, `interactions` has a shape of (731,927,5). It contains 731,927 reviews on recipes recorded in the dataset above. Columns are described in the following:

| Column       | Description                       |
|--------------|-----------------------------------|
| `user_id`    | User ID                           |
| `recipe_id`  | Recipe ID                         |
| `date`       | Date of interaction               |
| `rating`     | Rating given                      |
| `review`     | Review text                       |