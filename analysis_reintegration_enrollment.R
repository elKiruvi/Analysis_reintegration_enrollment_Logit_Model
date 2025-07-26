#-----------------------------------------------------------------------
# Part 1: Setup and Data Loading
#-----------------------------------------------------------------------

# Load necessary libraries for data manipulation and reading
# install.packages(c("readr", "dplyr", "knitr")) 
library(readr)
library(dplyr)
library(knitr) 

# --- Load the Dataset ---
# This dataset contains statistics on demobilized individuals in Colombia,
# tracking their socio-economic conditions and participation in the reintegration process.
# We will analyze factors like their former group, demographics, and benefits received.

setwd("C:/Users/Samuel Cardona Ochoa/Documents/GitHub/Analysis_reintegration_enrollment_Logit_Model")
df <- read_csv("ESTAD_STICAS_DE_LAS_PERSONAS_DESMOVILIZADAS_QUE_HAN_INGRESADO_AL_PROCESO_DE_REINTEGRACI_N_-_DEPURACI_N_DE_VARIABLES_20240517.csv")

# --- Initial Data Exploration ---

# Display the first few rows to understand the structure
head(df)

# Get a summary of the data, including column names and types
glimpse(df)

# Check the distribution of a key categorical variable, like education level
cat("Distribution of Education Levels:\n")
table(df$`Nivel Educativo`)

#-----------------------------------------------------------------------
# Part 2: Data Preparation and Feature Engineering
#-----------------------------------------------------------------------

# We use dplyr's pipe operator (%>%) for a clean workflow.
df_clean <- df %>%
  # Step 1: Select the variables of interest and rename them to English
  select(
    entered_program = `Ingreso/No ingreso`,
    former_group = `Ex Grupo`,
    demobilization_type = `Tipo de Desmovilizacion`,
    sex = `Sexo`,
    age_group = `Grupo Etario`,
    process_status = `Situacion Final frente al proceso`,
    benefit_trv = `BeneficioTRV`, # Benefit in the last year
    benefit_fa = `BeneficioFA`,   # Academic formation benefit
    benefit_fpt = `BeneficioFPT`, # Job training benefit
    benefit_pdt = `BeneficioPDT`, # Productive development benefit
    education_level = `Nivel Educativo`,
    disbursement_bie = `Desembolso BIE`,
    has_family_census = `Posee Censo de Familia?`,
    num_children = `N° de Hijos`,
    family_size = `Total Integrantes grupo familiar`
  ) %>%
  # Step 2: Create dummy variables from the categorical columns
  mutate(
    # Dependent Variable: 1 if they entered the program, 0 otherwise
    entered_program = ifelse(entered_program == "Sí", 1, 0),
    
    # Independent Variables (Dummies)
    demobilization_collective = ifelse(demobilization_type == "Colectiva", 1, 0),
    is_male = ifelse(sex == "Masculino", 1, 0),
    
    # Age group dummies (Reference group: "Over 40 years")
    age_18_25 = ifelse(age_group == "Entre 18 y 25 años", 1, 0),
    age_26_40 = ifelse(age_group == "Entre 26 y 40 años", 1, 0),
    
    # Process status dummies (Reference group: "Inactive/Other")
    process_completed = ifelse(process_status == "Culminado", 1, 0),
    in_process = ifelse(process_status == "En Proceso", 1, 0),
    
    # Benefit dummies
    benefit_trv = ifelse(benefit_trv == "Sí", 1, 0),
    benefit_fa = ifelse(benefit_fa == "Sí", 1, 0),
    benefit_fpt = ifelse(benefit_fpt == "Sí", 1, 0),
    benefit_pdt = ifelse(benefit_pdt == "Sí", 1, 0),
    
    # Education level dummies (Reference group: "No formal education")
    edu_literacy = ifelse(education_level == "Alfabetización", 1, 0),
    edu_primary = ifelse(education_level == "Básica Primaria", 1, 0),
    edu_secondary = ifelse(education_level == "Básica Secundaria", 1, 0),
    edu_highschool = ifelse(education_level == "Bachiller", 1, 0),
    
    disbursement_bie = ifelse(disbursement_bie == "Sí", 1, 0),
    has_family_census = ifelse(has_family_census == "Sí", 1, 0),
    
    # Former armed group dummies (Reference group: "Other groups")
    group_farc = ifelse(former_group == "FARC", 1, 0),
    group_auc = ifelse(former_group == "AUC", 1, 0)
  ) %>%
  # Step 3: Remove the original character columns that have been converted
  select(-c(former_group, demobilization_type, sex, age_group, process_status, education_level))

# --- Review the Cleaned Data ---
cat("\nSummary of the cleaned and prepared dataset:\n")
summary(df_clean)

#-----------------------------------------------------------------------
# Part 3: Model 1 - Full Logistic Regression
#-----------------------------------------------------------------------

# Estimate the logit model

full_logit_model <- glm(entered_program ~ ., 
                        data = df_clean,
                        family = binomial(link = "logit"))

# Print a comprehensive summary of the model
cat("\n--- Summary of the Full Logit Model ---\n")
summary(full_logit_model)
# The summary shows coefficients, standard errors, z-values, and p-values (Pr(>|z|)).
# A low p-value (e.g., < 0.05) suggests a statistically significant variable.
# We can see that several variables are not statistically significant in this full model.

# --- Model Fit Assessment ---

# 1. McFadden's Pseudo R-squared
# This is a goodness-of-fit measure for logistic regression, analogous to R-squared.
null_model <- glm(entered_program ~ 1, data = df_clean, family = binomial(link = "logit"))
pseudo_r2 <- 1 - as.vector(logLik(full_logit_model) / logLik(null_model))
cat(paste("\nMcFadden's Pseudo R-squared:", round(pseudo_r2, 4), "\n"))


# 2. Confusion Matrix
# This table shows how well the model's predictions match the actual outcomes.
predicted_values <- round(fitted(full_logit_model))
confusion_matrix <- table(Actual = df_clean$entered_program, 
                          Predicted = predicted_values)

cat("\nConfusion Matrix:\n")
print(confusion_matrix)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat(paste("\nModel Accuracy:", round(accuracy, 4), "\n"))


# 3. ROC Curve and AUC
# The ROC curve plots the true positive rate against the false positive rate.
# The Area Under the Curve (AUC) measures the model's ability to discriminate between classes.
# An AUC of 0.5 is random chance, while 1.0 is a perfect classifier.
# install.packages("pROC") 

library(pROC)
roc_curve <- roc(df_clean$entered_program, fitted(full_logit_model))
plot(roc_curve, main = "ROC Curve for Full Logit Model", col = "blue", lwd = 2)
abline(a=0, b=1, lty=2, col="gray") # Add diagonal line for reference
legend("bottomright", legend=paste("AUC =", round(auc(roc_curve), 4)), bty="n")
# An AUC of ~0.95 indicates excellent discriminatory power.

#-----------------------------------------------------------------------
# Part 4: Model Simplification using AIC
#-----------------------------------------------------------------------
# install.packages("MASS") 
library(MASS)

# stepAIC automatically adds/removes variables to find the model with the lowest AIC.
# A lower AIC indicates a better model in terms of fit versus complexity.
cat("\n--- Performing Stepwise Selection using AIC ---\n")
parsimonious_model_selection <- stepAIC(full_logit_model, direction = "backward", trace = FALSE)

# Show the variables selected for the final, more parsimonious model
cat("\nVariables selected by AIC for the final model:\n")
print(names(parsimonious_model_selection$coefficients))
# AIC suggests removing variables that do not contribute significantly to the model's fit,
# such as benefit_fa, benefit_fpt, and num_children in the original analysis.

#-----------------------------------------------------------------------
# Part 5: Model 2 - The Parsimonious Logistic Regression
#-----------------------------------------------------------------------

# Estimate the second, simpler logit model using the formula from the AIC selection
final_logit_model <- glm(formula(parsimonious_model_selection), 
                         data = df_clean,
                         family = binomial(link = "logit"))

# Print the summary of the final model
cat("\n--- Summary of the Final (Parsimonious) Logit Model ---\n")
summary(final_logit_model)
# We observe that this model is more efficient. Most variables are statistically significant,
# and it is easier to interpret.

# --- Re-evaluate Model Fit for the Final Model ---

# 1. McFadden's Pseudo R-squared
pseudo_r2_final <- 1 - as.vector(logLik(final_logit_model) / logLik(null_model))
cat(paste("\nFinal Model - McFadden's Pseudo R-squared:", round(pseudo_r2_final, 4), "\n"))
# The Pseudo R-squared barely decreases, which is excellent. We removed variables without losing much explanatory power.

# 2. Confusion Matrix
predicted_values_final <- round(fitted(final_logit_model))
confusion_matrix_final <- table(Actual = df_clean$entered_program, 
                                Predicted = predicted_values_final)

cat("\nFinal Model - Confusion Matrix:\n")
print(confusion_matrix_final)
accuracy_final <- sum(diag(confusion_matrix_final)) / sum(confusion_matrix_final)
cat(paste("\nFinal Model Accuracy:", round(accuracy_final, 4), "\n"))
# The accuracy remains nearly identical, meaning our simpler model predicts just as well.

# 3. ROC Curve and AUC
roc_curve_final <- roc(df_clean$entered_program, fitted(final_logit_model))
plot(roc_curve_final, main = "ROC Curve for Final Logit Model", col = "darkgreen", lwd = 2)
abline(a=0, b=1, lty=2, col="gray")
legend("bottomright", legend=paste("AUC =", round(auc(roc_curve_final), 4)), bty="n")
# The AUC is virtually unchanged, confirming our model's high discriminatory power.

# --- Conclusion on Model Selection ---
# The final model is statistically superior because it is more parsimonious (has more degrees of freedom)
# while maintaining the same level of predictive accuracy and discriminatory power as the full model.

#-----------------------------------------------------------------------
# Part 6: Robustness Checks and Visualization
#-----------------------------------------------------------------------
# install.packages(c("lmtest", "sandwich", "ggplot2")) 
library(lmtest)
library(sandwich)
library(ggplot2)

# --- Robustness Check: Heteroskedasticity-Consistent Standard Errors ---
# Standard errors in logit models can be biased if heteroskedasticity is present.
# We can calculate "robust" standard errors to get more reliable p-values.
robust_summary <- coeftest(final_logit_model, vcov = vcovHC(final_logit_model, type = "HC1"))
cat("\n--- Final Model with Robust Standard Errors ---\n")
print(robust_summary)
# Comparing this table to the standard summary shows if our significance levels are
# robust to potential heteroskedasticity.

# --- Professional Visualization using ggplot2 ---
# Let's visualize the relationship between family size and program enrollment.
# A boxplot is excellent for this.

# First, we need to make the dependent variable a factor for plotting
df_clean$entered_program_factor <- factor(df_clean$entered_program,
                                          levels = c(0, 1),
                                          labels = c("Did Not Enter", "Entered Program"))

ggplot(df_clean, aes(x = entered_program_factor, y = family_size, fill = entered_program_factor)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    title = "Program Enrollment by Family Size",
    subtitle = "Distribution of family members for enrollees vs. non-enrollees",
    x = "Program Enrollment Status",
    y = "Total Family Members"
  ) +
  theme_minimal() +
  theme(legend.position = "none") # The x-axis is already labeled

#-----------------------------------------------------------------------
# Part 7: Interpretation and Conclusion
#-----------------------------------------------------------------------

# --- Calculating and Interpreting Odds Ratios ---
# An odds ratio > 1 means the event is more likely as the predictor increases.
# An odds ratio < 1 means the event is less likely as the predictor increases.
# An odds ratio = 1 means the predictor has no effect on the odds.

odds_ratios <- exp(coef(final_logit_model))

# Combine coefficients, robust standard errors, and odds ratios into a single table
final_summary_table <- data.frame(
  Coefficient = coef(final_logit_model),
  Robust_Std_Error = robust_summary[, "Std. Error"],
  Robust_P_Value = robust_summary[, "Pr(>|z|)"],
  Odds_Ratio = odds_ratios
)

cat("\n--- Final Model Summary with Odds Ratios ---\n")
# Using kable for a nicely formatted table
knitr::kable(final_summary_table, digits = 4, caption = "Final Model Results")
