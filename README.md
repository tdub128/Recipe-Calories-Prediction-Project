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

## Data Cleaning and Exploratory Data Analysis
### Data Cleaning Steps

To make my analysis of the dataset more efficient and convenient, I conducted the following data cleaning steps:

---

**1. Left Merge the `recipes` and `interactions` Datasets on `id` and `recipe_id`.**  
This step matches unique recipes with their ratings from user interactions. Review column is dropped since it is lengthy and has little relevance to the project. By merging these datasets, I enriched the recipe dataset with user feedback, which is crucial for understanding recipe popularity and quality.

---

**2. Fill All Ratings of `0` with `np.nan`.**  
Ratings ranges from 1 to 5, where 1 indicates the lowest rating and 5 indicates the highest rating. A rating of `0` means the user didn't perform the rating action, therefore is invalid and signifies missing data.

---

**3. Add a Column `average_rating` Containing the Average Rating Per Recipe.**  
Since a recipe can receive multiple ratings from different users, I computed the average rating for each recipe. This provides a more comprehensive and representative understanding of the quality of a given recipe. Incorporating this feature aids in exploring correlations between recipe rating and its nutritional components.

---

**4. Split Values in the `nutrition` Column into Individual Columns of Floats.**  
Although the `nutrition` column appears as a list, it is actually stored as a string object. Based on the dataset documentation, I know the meaning of each value inside the brackets. By applying a lambda function, I split the values into individual columns (e.g., `calories`, `sugar (%)`) and converted them to floats. This is crucial for enabling numerical analysis and preparing the response variable (`sugar (%)`) for prediction.

---

Here are the head of the dataframe we used for our analysis. The cleaned dataframe has 83,782 rows and 12 columns. To keep the website clean, only columns used for the analysis are included. Note that `steps` is also a feature used in the prediction model, it is not shown here because it contains very long strings.

| name                                 | tags                                                                                                                                                                                                                                                                                               |   minutes | description                                                                                                                                                                                                                                                                                                                                                                       | ingredients                                                                                                                                                                                                                             | nutrition                                                   |   n_steps |   sugar (%) |   rating_avg |
|:-------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------|----------:|------------:|-------------:|
| 1 brownies in the world    best ever | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings']                                                                        |        40 | these are the most; chocolatey, moist, rich, dense, fudgy, delicious brownies that you'll ever make.....sereiously! there's no doubt that these will be your fav brownies ever for you can add things to them or make them plain.....either way they're pure heaven!                                                                                                              | ['bittersweet chocolate', 'unsalted butter', 'eggs', 'granulated sugar', 'unsweetened cocoa powder', 'vanilla extract', 'brewed espresso', 'kosher salt', 'all-purpose flour']                                                          | ['138.4', '10.0', '50.0', '3.0', '3.0', '19.0', '6.0']      |        10 |          50 |            4 |
| 1 in canada chocolate chip cookies   | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                                                                                                      |        45 | this is the recipe that we use at my school cafeteria for chocolate chip cookies. they must be the best chocolate chip cookies i have ever had! if you don't have margarine or don't like it, then just use butter (softened) instead.                                                                                                                                            | ['white sugar', 'brown sugar', 'salt', 'margarine', 'eggs', 'vanilla', 'water', 'all-purpose flour', 'whole wheat flour', 'baking soda', 'chocolate chips']                                                                             | ['595.1', '46.0', '211.0', '22.0', '13.0', '51.0', '26.0']  |        12 |         211 |            5 |
| 412 broccoli casserole               | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                                                                                               |        40 | since there are already 411 recipes for broccoli casserole posted to "zaar" ,i decided to call this one  #412 broccoli casserole.i don't think there are any like this one in the database. i based this one on the famous "green bean casserole" from campbell's soup. but i think mine is better since i don't like cream of mushroom soup.submitted to "zaar" on may 28th,2008 | ['frozen broccoli cuts', 'cream of chicken soup', 'sharp cheddar cheese', 'garlic powder', 'ground black pepper', 'salt', 'milk', 'soy sauce', 'french-fried onions']                                                                   | ['194.8', '20.0', '6.0', '32.0', '22.0', '36.0', '3.0']     |         6 |           6 |            5 |
| millionaire pound cake               | ['time-to-make', 'course', 'cuisine', 'preparation', 'occasion', 'north-american', 'desserts', 'american', 'southern-united-states', 'dinner-party', 'holiday-event', 'cakes', 'dietary', 'christmas', 'thanksgiving', 'low-sodium', 'low-in-something', 'taste-mood', 'sweet', '4-hours-or-less'] |       120 | why a millionaire pound cake?  because it's super rich!  this scrumptious cake is the pride of an elderly belle from jackson, mississippi.  the recipe comes from "the glory of southern cooking" by james villas.                                                                                                                                                                | ['butter', 'sugar', 'eggs', 'all-purpose flour', 'whole milk', 'pure vanilla extract', 'almond extract']                                                                                                                                | ['878.3', '63.0', '326.0', '13.0', '20.0', '123.0', '39.0'] |         7 |         326 |            5 |
| 2000 meatloaf                        | ['time-to-make', 'course', 'main-ingredient', 'preparation', 'main-dish', 'potatoes', 'vegetables', '4-hours-or-less', 'meatloaf', 'simply-potatoes2']                                                                                                                                             |        90 | ready, set, cook! special edition contest entry: a mediterranean flavor inspired meatloaf dish. featuring: simply potatoes - shredded hash browns, egg, bacon, spinach, red bell pepper, and goat cheese.                                                                                                                                                                         | ['meatloaf mixture', 'unsmoked bacon', 'goat cheese', 'unsalted butter', 'eggs', 'baby spinach', 'yellow onion', 'red bell pepper', 'simply potatoes shredded hash browns', 'fresh garlic', 'kosher salt', 'white pepper', 'olive oil'] | ['267.0', '30.0', '12.0', '12.0', '29.0', '48.0', '2.0']    |        17 |          12 |            5 |

---
ohhhhhh
