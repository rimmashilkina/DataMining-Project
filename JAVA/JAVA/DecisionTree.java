import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.ArrayList;
import java.lang.Math;
import java.util.Collection;
import java.util.Comparator;

//main class
public class DecisionTree {
	public static void main(String[] args) {
		//importing CSV file
		DataFrame df_train = new DataFrame("cs_train.csv");
		DataFrame df_test = new DataFrame("cs_test.csv");
      
		//print dataframe
		df_train.printDataFrame("Train");

		//print header - do not print it in train as train is recursive
		System.out.println("\nDecisionFeature:\tThreshold:\tmaxInfoGain");

		//creating a structure for a decision tree
		DecisionTreeNode node = new DecisionTreeNode(df_train);

		//Generate DecisionTree model
		node.train();

		//testing our tree.
		df_test.printDataFrame("Test");
		node.test(df_test);

		//validation - another csv file with values
		DataFrame tdf = new DataFrame("tcs_data.csv");
		node.test(tdf);
	}
}

// structure to store values from a CSV file
class DataFrame {
	private List<Entity> cs_data;

	//constructor for the class
	//reads a file into the list
	//filename as argument
	DataFrame(String filename) {
		cs_data = new ArrayList<>();

		//reading file line by line and adding them into the list (cs_data)
		Scanner fin;
		try {
			fin = new Scanner(new File(filename));
			while (fin.hasNext()) {
				String inputLine = fin.nextLine();
				String[] items = inputLine.split(",");
				Entity temp = new Entity();
				for (int i = 0; i < items.length - 1; i++) {
					temp.addFeature(Double.valueOf(items[i]));
				}
				temp.setLabel(items[items.length - 1]);
				this.append(temp);
			}
			fin.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	//empty constructor for the class
	public DataFrame() {
		cs_data = new ArrayList<>();
	}

	//return an item at index from the list
	public Entity getEntity(int index) {
		return cs_data.get(index);
	}

	//perform split of the list at some index point
	public List<DataFrame> split(int splitPoint)
	{
		DataFrame part0 = new DataFrame();
		DataFrame part1 = new DataFrame();
		for (int i = 0; i < cs_data.size(); i++) {
			if (i <= splitPoint) {
				part0.append(cs_data.get(i));
			} else {
				part1.append(cs_data.get(i));
			}
		}
		List<DataFrame> list = new ArrayList<>();
		list.add(part0);
		list.add(part1);
		return list;
	}


	public List<Double> getFeatureList(int feature) {
		List<Double> featureList = new ArrayList<>();
		for (Entity e : cs_data)
		{
			featureList.add(e.getFeature(feature));
		}
		return featureList;
	}

	//add entity into the list
	public void append(Entity row) {
		cs_data.add(row);
	}

	//return number of features in a single line
	public int getFeatureNo() {
		return cs_data.get(0).getFeatureNo();
	}

	//return number of entities in the list
	public int getEntityCount() {
		return cs_data.size();
	}

	//find out a number of distinct labels
	public Map<String, Integer> getLabelCount() {
		Map<String, Integer> labelCount = new HashMap<>();
		for (Entity e : cs_data) {
			Integer previousValue = labelCount.get(e.getLabel());
			labelCount.put(e.getLabel(), previousValue == null ? 1 : previousValue + 1);
		}
		return labelCount;
	}

	//sort dataframe
	public DataFrame sortBy(int index) {
		Collections.sort(cs_data, new SortEntity(index));
		return this;
	}

	// Construct a string (i.e. for printing)
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("[\n");
		for (Entity e : cs_data) {
			sb.append(e);
			sb.append(",\n");
		}
		sb.append("]");
		return sb.toString();
	}

	public void printDataFrame(String label) {
		System.out.println("Printing Dataframe: " + label);
		for (Entity e : cs_data) {
			System.out.println(e.toString());
		}
	}
}

class DecisionTreeNode {
	private transient DataFrame cs_data;
	private double cs_threshold;
	private int cs_decisionFeature;
	private DecisionTreeNode cs_leftChild;
	private DecisionTreeNode cs_rightChild;
	private int cs_depth;
	private String cs_decision;
	private boolean cs_isLeft;

	//constructor or DecisionTreeNode class
	public DecisionTreeNode(DataFrame data) {
		cs_data = data;
		cs_depth = 0;
	}

	//constructor for DecisionTreeNode
	//parameters: Data, depth of the node, indicator if the node is Left (true) or Right (false)
	public DecisionTreeNode(DataFrame data, int depth, boolean isLeft) {
		cs_data = data;
		cs_depth = depth;
		cs_isLeft = isLeft;
	}


	public void train() {
		//set number of independent variables / features
		int nFeature = cs_data.getFeatureNo();

		//create labels' map
		Map<String, Integer> originalCount = cs_data.getLabelCount();

		//calculate entropy for data labels
		double originalEntropy = Probability.entropy(originalCount.values());

		//set initial value for maxInfoGain
		double maxInfoGain = 0;

		//set initial splitpoint. -1 indicates no split
		int splitPoint = -1;

		//Search for threshold among all features
		for (int ithFeature = 0; ithFeature < nFeature; ithFeature++) {
			Map<String, Integer> countPart1 = new HashMap<>();
			Map<String, Integer> countPart2 = new HashMap<>();
			for (String key : originalCount.keySet()) {
				countPart2.put(key, originalCount.get(key));
			}
			cs_data.sortBy(ithFeature);

			// Search for threshold in this data set
			for (int jthEntity = 0; jthEntity < cs_data.getEntityCount() - 1; jthEntity++) {
				Entity currentEntity = cs_data.getEntity(jthEntity);
				Entity nextEntity = cs_data.getEntity(jthEntity + 1);
				double currentFeature = currentEntity.getFeature(ithFeature);
				String currentLabel = currentEntity.getLabel();
				double nextFeature = nextEntity.getFeature(ithFeature);

				// Update count
				Integer count = countPart1.get(currentLabel);
				countPart1.put(currentLabel, count == null ? 1 : count + 1);
				countPart2.put(currentLabel, countPart2.get(currentLabel) - 1);

				if (currentFeature != nextFeature) {
					Double newEntropy = Probability.entropy(countPart1.values(), countPart2.values());
					Double infoGain = originalEntropy - newEntropy;
					if (infoGain > maxInfoGain) {
						maxInfoGain = infoGain;
						cs_decisionFeature = ithFeature;
						splitPoint = jthEntity;
						cs_threshold = (currentEntity.getFeature(ithFeature) + nextEntity.getFeature(ithFeature)) / 2.0;
					}
				}
			}
		}

		if (splitPoint == -1) {
			cs_decision = cs_data.getEntity(0).getLabel();
		} else {
			System.out.println(String.format("%d, %f, %f", cs_decisionFeature, cs_threshold, maxInfoGain));
			cs_data.sortBy(cs_decisionFeature);
			List<DataFrame> chunks = cs_data.split(splitPoint);
			cs_leftChild = new DecisionTreeNode(chunks.get(0), cs_depth + 1, true);
			cs_leftChild.train();
			cs_rightChild = new DecisionTreeNode(chunks.get(1), cs_depth + 1, false);
			cs_rightChild.train();
		}
	}

	//test DecisionTree for an entity
	public String test(Entity e) {
		if (cs_decision != null) {
			return cs_decision;
		} else if (e.getFeature(cs_decisionFeature) <= cs_threshold) {
			return cs_leftChild.test(e);
		} else {
			return cs_rightChild.test(e);
		}
	}

	//run test of the DecisionTree for a full dataflrame (i.e. CSV file)
	public List<String> test(DataFrame df) {
		List<String> nl = new ArrayList<>();
		//print predictions and actual data labels
		System.out.println("\nPrediction:\tActual Label");

		//test each entity and see if labels match
		for (int i = 0; i < df.getEntityCount(); i++) {
			String prediction = test(df.getEntity(i));
			nl.add(prediction);
			System.out.println(String.format("%s:\t\t\t%s", prediction, df.getEntity(i).getLabel()));
		}
		return nl;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();

		for (int i = 0; i < cs_depth - 1; i++) {
			sb.append(' ');
		}
		if (cs_depth > 0) {
			sb.append("L" + (cs_isLeft ? "<" : ">"));
		}

		if (cs_leftChild == null && cs_rightChild == null) {
			sb.append(cs_decision + "\n");
		} else {
			sb.append(String.format("Feature %d split on %f \n", cs_decisionFeature, cs_threshold));
			sb.append(cs_leftChild.toString());
			sb.append(cs_rightChild.toString());
		}
		return sb.toString();
	}
}

class Entity {
	List<Double> cs_data;
	String cs_label;

	//constructor
	//cs_data is new (empty) ArrayList
	public Entity() {
		cs_data = new ArrayList<>();
	}

	//add feature/independent variable into the entity
	public Entity addFeature(double value) {
		cs_data.add(value);
		return this;
	}

	//set label for an entity
	public Entity setLabel(String label) {
		cs_label = label;
		return this;
	}

	//return number of elements (features / independent variables) in an entity
	public int getFeatureNo() {
		return cs_data.size();
	}

	//return value of a specific variable / feature in an entity
	public double getFeature(int feature) {
		return cs_data.get(feature);
	}

	//return label assigned to an entity
	public String getLabel() {
		return cs_label;
	}

	//return values of variables and a label of an entity as a string
	@Override
	public String toString() {
		return cs_data + "\t-\t" + cs_label;
	}
}

class Probability {

	//calculate entropy for a single set
	public static double entropy(Collection<Integer> counts) {
		int sum = counts.stream().mapToInt(Integer::intValue).sum();
		double entropy = 0;
		for (Integer count : counts) {
			if (count > 0) {
				double p = count.doubleValue() / sum;
				entropy -= p * Math.log(p);
			}
		}
		return entropy;
	}

	//calculate entropy for two sets
	public static double entropy(Collection<Integer> part1, Collection<Integer> part2) {
		int sumPart1 = part1.stream().mapToInt(Integer::intValue).sum();
		int sumPart2 = part2.stream().mapToInt(Integer::intValue).sum();
		int sum = sumPart1 + sumPart2;
		return sumPart1 * entropy(part1) / sum + sumPart2 * entropy(part2) / sum;
	}
}

class SortEntity implements Comparator<Entity> {
	int cs_idx;

	public SortEntity(int idx) {
		cs_idx = idx;
	}

	//compare two entities
	@Override
	public int compare(Entity o1, Entity o2) {
		double f1 = o1.getFeature(cs_idx);
		double f2 = o2.getFeature(cs_idx);
		if (f1 < f2) return -1;
		if (f1 > f2) return 1;
		return o1.getLabel().compareTo(o2.getLabel());
	}

}