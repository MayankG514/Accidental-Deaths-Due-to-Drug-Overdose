library(conflicted)
library(tidyverse)
library(ggplot2)
library(gridExtra)
library(grid)
conflicts_prefer(dplyr::filter)
# Reading data from the AWS S3 bucket
data = read.csv('https://mayankgroverawsbucket.s3.amazonaws.com/Accidental_Drug_Related_Deaths_2012-2022.csv')

data_CT = data[data$Death.State=='CT',]

sum(is.na(data_CT$Age))
data_CT <- na.omit(data_CT)

# Plotting the histogram for the age variable 
ggplot(data_CT, aes(x = Age)) +
  geom_histogram(bins = 25, aes(y = ..density..)) +
  geom_density(alpha = 0.2, fill = "blue") +
  labs(title = 'Density plot of age distribution', x='Age', y='Density' )

# Plotting the bar graphs to study the gender ratio 
unique(data_CT$Sex)
data_CT$Sex <- replace(data_CT$Sex, data_CT$Sex == "", "Other")

ggplot(data_CT, aes(x = Sex)) +
  geom_bar(fill = c('cyan','lightblue', 'orange'))+
  labs(title = 'Gender distribution in accidental deaths observed', x='Gender', y='Count' )

unique(data_CT$Race)
data_CT$Race <- replace(data_CT$Race, data_CT$Race == "", "Unknown")
unique(data_CT$Race)

ggplot(data_CT, aes(x = Race)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  geom_bar()

# From the above plot we observe that most of the accidental deaths occured prominently 
# in the following three races : 
# White
# Black
# Black or African American

# Get counts using table
counts <- table(data_CT$Death.City)

# Create a data frame with unique values and counts
result <- data.frame(City = names(counts), Count = as.numeric(counts))
result2 <- result[order(result$Count, decreasing = TRUE), ] 
subset_data <- result2 %>% filter(result2$Count > 100)

# Plotting the cities with more than 100 deaths due to drug overdose
ggplot(subset_data, aes(x=City,y=Count, fill=City))+geom_bar(stat='identity')+
  theme(axis.text.x = element_text(size = 7, angle = 90, vjust = 0.5, hjust=1),
        legend.position = 'none')+
  labs(title='City wise death counts in Connecticut')

# Extracting the year from the date column and plotting it using a line chart
data_CT$Year <- substring(data_CT$Date,7,10)
count_year <- as.data.frame(table(data_CT$Year))

# Rename the columns for clarity
colnames(count_year) <- c("Year", "Count")

ggplot(count_year, aes(x = Year, y = Count, group = 1)) +
  geom_line() +
  geom_point() +
  labs(title = "Year wise death counts due to drug overdose",
       x = "Year",
       y = "Count")


 # write.csv(data_CT, "data_CT.csv", row.names=FALSE)
