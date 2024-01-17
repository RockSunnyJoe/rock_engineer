package main

import (
	"errors"
	"fmt"
	"math"
)

var sum int

func init() {
	sum = 2
}

func longestOnes(nums []int, k int) int {
	/*
	   对于任意的左端点，希望找到最大的右端点，使得 [left,right]包含不超过 k 个 0。
	   只要我们枚举所有可能的左端点，将得到的区间的长度取最大值，即可得到答案。
	*/
	var maxOneCnt = 0
	for i := 0; i < len(nums); i++ {
		zeroCnt := 0
		j := i
		for ; j < len(nums); j++ {
			if nums[j] == 1 {
				continue
			}
			zeroCnt++
			if zeroCnt >= k {
				if maxOneCnt < j-i {
					maxOneCnt = j - i
				}
			}
			fmt.Println(i, j, zeroCnt)
		}
	}
	return maxOneCnt
}

func reverseWords(s string) string {
	// 识别单词放到堆栈里面
	var stack [][]byte
	strs := []byte(s)

	var preBytes []byte
	for _, v := range strs {
		if v == ' ' {
			if len(preBytes) != 0 {
				stack = append([][]byte{preBytes}, stack...)
				preBytes = []byte{}
			}
			continue
		}
		preBytes = append(preBytes, v)
	}
	if len(preBytes) != 0 {
		stack = append([][]byte{preBytes}, stack...)
		preBytes = []byte{}
	}
	var ret string
	for i, v := range stack {
		if i > 0 && i < len(stack)-1 {
			ret = fmt.Sprintf("%s %s ", ret, string(v))
		} else if i == 0 {
			ret = fmt.Sprintf("%s", string(v))
		} else if i == len(stack)-1 {
			ret = fmt.Sprintf("%s %s", ret, string(v))
		}
	}
	return ret
}

/*

select t.restaurant_id, t.dish_name, t.dish_price
from t
right join (
select max(dish_price) as max_price, restaurant_id
from t
group by restaurant_id) as sub
on sub.max_price = t.dish_price and sub.restaurant_id = t.restaurant_id

WITH RankedData AS (
    SELECT
        your_column,
        ROW_NUMBER() OVER (ORDER BY your_column) AS RowAsc,
        ROW_NUMBER() OVER (ORDER BY your_column DESC) AS RowDesc
    FROM your_table
)
SELECT AVG(your_column) AS median
FROM RankedData
WHERE RowAsc = RowDesc OR RowAsc + 1 = RowDesc OR RowAsc = RowDesc + 1;

*/

func numIslands(grid [][]string) int {
	m := len(grid)
	n := len(grid[0])
	cnt := 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == "1" {
				bsfIsland(grid, i, j)
				fmt.Println(grid)
				cnt++
			}
		}
	}
	return cnt
}

type GraphNode struct {
	Name         string
	Neighbor     []string
	NeighborCost map[string]float64
}

func bsfIsland(grid [][]string, m, n int) {
	stack := [][]int{}
	stack = append(stack, []int{m, n})
	for len(stack) > 0 {
		x := stack[0][0]
		y := stack[0][1]
		grid[x][y] = "2"
		if x-1 >= 0 && grid[x-1][y] == "1" {
			stack = append(stack, []int{x - 1, y})
			grid[x-1][y] = "2"
		}
		if x+1 < len(grid) && grid[x+1][y] == "1" {
			stack = append(stack, []int{x + 1, y})
			grid[x+1][y] = "2"
		}
		if y-1 >= 0 && grid[x][y-1] == "1" {
			stack = append(stack, []int{x, y - 1})
			grid[x][y-1] = "2"
		}
		if y+1 < len(grid[0]) && grid[x][y+1] == "1" {
			stack = append(stack, []int{x, y + 1})
			grid[x][y+1] = "2"
		}
		if len(stack) == 1 {
			stack = [][]int{}
		} else {
			stack = stack[1:]
		}
	}
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

var ret []string

func letterCombinations(digits string) []string {
	inputArray := []byte(digits)

	ret = []string{}
	getCombines(inputArray, 0, []byte{})
	return ret
}

func getCombines(array []byte, index int, combine []byte) {
	if index == len(array) {
		ret = append(ret, string(combine))
		return
	}

	mapArray := getMapArray(array[index])
	for _, v := range mapArray {
		combine = append(combine, v)
		getCombines(array, index+1, combine)
		combine = combine[:len(combine)-1]
	}
}

var badWords map[string]int

type WordsTree struct {
	Val           byte
	IsWord        bool
	ChildrenNodes []*WordsTree
}

func generateWordsTree(words []string, root *WordsTree) {
	for _, v := range words {
		insertToWordsTree(v, root)
	}
}

func insertToWordsTree(word string, root *WordsTree) {
	chars := []byte(word)

	currentNode := root

	for _, c := range chars {
		if currentNode.ChildrenNodes[c-'a'] == nil {
			currentNode.ChildrenNodes[c-'a'] = &WordsTree{
				Val:           c,
				ChildrenNodes: make([]*WordsTree, 26),
			}
			//fmt.Println(string(c))
		}
		currentNode = currentNode.ChildrenNodes[c-'a']
	}
}

func searchWordsTree(word string, root *WordsTree) bool {
	chars := []byte(word)
	currentNode := root
	for _, c := range chars {
		if currentNode.ChildrenNodes[c-'a'] == nil {
			return false
		} else {
			if currentNode.ChildrenNodes[c-'a'].IsWord {
				return true
			}
			currentNode = currentNode.ChildrenNodes[c-'a']
		}
	}
	return true
}

func checkIsBad(input string) bool {

	array := []byte(input)

	for i := 0; i < len(array); i++ {
		for j := i; j < len(array); j++ {
			if checkSubString(string(array[i : j+1])) {
				return true
			}
		}
	}
	return false
}

func checkSubString(words string) bool {
	if badWords[words] == 1 {
		return true
	}
	return false
}

func find(root *Node, left, right int) int {
	if root == nil {
		return -1
	}
	if root.Val == left || root.Val == right {
		return root.Val
	}

	leftFind := find(root.Left, left, right)
	rightFind := find(root.Right, left, right)

	if leftFind != -1 && rightFind != -1 {
		return root.Val
	}
	if leftFind != -1 {
		return leftFind
	} else {
		return rightFind
	}
	return -1
}

func minimumTotal(triangle [][]int) int {
	miniTotal := make([][]int, len(triangle))
	miniOne := math.MaxInt
	for i, rows := range triangle {
		miniTotal[i] = make([]int, len(rows))
		for j, v := range rows {
			if i == 0 {
				miniTotal[0][j] = v
				continue
			}
			if j == 0 {
				miniTotal[i][j] = miniTotal[i-1][j] + v
				continue
			}
			if j < len(miniTotal[i-1]) {
				miniTotal[i][j] = getMin(miniTotal[i-1][j], miniTotal[i-1][j-1]) + v
			} else {
				miniTotal[i][j] = miniTotal[i-1][j-1] + v
			}
		}
	}

	// for last row, log the minimal
	for j, v := range miniTotal[len(miniTotal)-1] {
		if j == 0 {
			miniOne = v
		} else if miniOne > v {
			miniOne = v
		}
	}
	return miniOne
}

func getMin(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func getMapArray(c byte) []byte {
	switch c {
	case '2':
		return []byte{'a', 'b', 'c'}
	case '3':
		return []byte{'d', 'e', 'f'}
	}
	return nil
}

func groupAnagrams(strs []string) [][]string {
	sortedMap := map[string][]int{}

	for index, v := range strs {
		sortedV := getSorted(v)
		existArray := sortedMap[sortedV]
		if existArray == nil {
			sortedMap[sortedV] = []int{index}
		} else {
			sortedMap[sortedV] = append(sortedMap[sortedV], index)
		}
	}

	output := [][]string{}
	for _, sameVs := range sortedMap {
		sameArray := []string{}
		for _, one := range sameVs {
			sameArray = append(sameArray, strs[one])
		}
		output = append(output, sameArray)
	}
	return output
}

func getSorted(v string) string {
	chars := []byte(v)
	cntMap := map[byte]int{}
	for _, v := range chars {
		cntMap[v]++
	}

	output := []byte{}
	for i := 0; i < 26; i++ {
		b := byte('a' + i)
		for i := 0; i < cntMap[b]; i++ {
			output = append(output, b)
		}
	}
	return string(output)
}
func moveZeroes(nums []int) {
	cnt := 0
	for _, v := range nums {
		if v == 0 {
			cnt++
		}
	}
	if cnt == 0 {
		return
	}
	fmt.Println(cnt)
	leftIndex := 0
	for _, v := range nums {
		if v != 0 {
			nums[leftIndex] = v
			leftIndex++
		}
	}

	for i := len(nums) - cnt; i < len(nums); i++ {
		nums[i] = 0
	}
}

func uniquePathsWithObstacles(obstacleGrid [][]int) int {
	sum = 0
	visited := make([]map[int]bool, len(obstacleGrid))
	for i, _ := range obstacleGrid {
		visited[i] = map[int]bool{}
	}
	tryPath(obstacleGrid, 0, 0, visited)
	return sum
}

func tryPath(obstacleGrid [][]int, i, j int, visited []map[int]bool) {
	if (i < 0) || (i >= len(obstacleGrid)) || (j < 0) || (j >= len(obstacleGrid[0])) {
		return
	}
	if (i == len(obstacleGrid)-1) && (j == len(obstacleGrid[0])-1) {
		sum++
		return
	}
	if visited[i][j] || obstacleGrid[i][j] == 1 {
		return
	}

	visited[i][j] = true
	tryPath(obstacleGrid, i+1, j, visited)
	tryPath(obstacleGrid, i-1, j, visited)
	tryPath(obstacleGrid, i, j-1, visited)
	tryPath(obstacleGrid, i, j+1, visited)
}

/*
equations := [][]string{
		[]string{"a", "b"},
		[]string{"e", "f"},
		[]string{"b", "e"}}
	values := []float64{3.4, 1.4, 2.3}
	queries := [][]string{
		[]string{"b", "a"},
		[]string{"a", "f"}}
	//[["b","a"],["a","f"],["f","f"],["e","e"],["c","c"],["a","c"],["f","e"]]
	num := calcEquation(equations, values, queries)
	fmt.Println(num)
*/
func calcEquation(
	equations [][]string,
	values []float64,
	queries [][]string) (ret []float64) {

	nodeMap := map[string]*GraphNode{}

	for i, v := range equations {
		v1 := v[0]
		v2 := v[1]

		if nodeMap[v1] == nil {
			nodeMap[v1] = createNode(v1)
		}
		if nodeMap[v2] == nil {
			nodeMap[v2] = createNode(v2)
		}

		nodeMap[v1].Neighbor = append(nodeMap[v1].Neighbor, v2)
		nodeMap[v1].NeighborCost[v2] = values[i]

		nodeMap[v2].Neighbor = append(nodeMap[v2].Neighbor, v1)
		nodeMap[v2].NeighborCost[v1] = 1 / values[i]
	}

	fmt.Println(nodeMap["a"])
	fmt.Println(nodeMap["b"])
	fmt.Println(nodeMap["e"])
	fmt.Println(nodeMap["f"])
	// a b e f
	for _, v := range queries {
		if v[0] == v[1] {
			ret = append(ret, 1)
			continue
		}
		path := findPath(nodeMap, v[0], v[1])
		fmt.Println(v, path)
		if len(path) == 0 {
			ret = append(ret, -1)
			continue
		}
		from := path[0]
		sum := float64(1)
		for i, road := range path {
			if i == 0 {
				continue
			}
			sum *= nodeMap[from].NeighborCost[road]
			from = road
		}
		ret = append(ret, sum)
	}

	return
}

func createNode(Name string) *GraphNode {
	return &GraphNode{Name: Name, NeighborCost: map[string]float64{}}
}

var roads []string

func findPath(nodeMap map[string]*GraphNode, start, end string) []string {
	if nodeMap[start] == nil || nodeMap[end] == nil {
		return nil
	}

	roads = []string{start}
	searched := map[string]bool{start: true}
	findPathSub(nodeMap, start, end, searched)
	return roads
}

func findPathSub(nodeMap map[string]*GraphNode, start, dest string, searched map[string]bool) bool {
	if start == dest {
		return true
	}
	for _, v := range nodeMap[start].Neighbor {
		fmt.Println("findPathSub", v, dest)
		if searched[v] {
			continue
		}
		searched[v] = true
		roads = append(roads, v)
		if findPathSub(nodeMap, v, dest, searched) {
			return true
		}
		roads = roads[:len(roads)-1] //remove last
	}
	return false
}

func getInOrderArray(root *TreeNode) []int {
	if root == nil {
		return nil
	}
	left := getInOrderArray(root.Left)
	if left != nil {
		left = append(left, root.Val)
	} else {
		left = []int{root.Val}
	}
	right := getInOrderArray(root.Right)
	left = append(left, right...)
	return left
}

func test1() error {
	return errors.New("test2")
}

var cnt int

func test2() error {
	//fmt.Println("execute")
	cnt++
	return nil
}

func caculateBalance(root *Node, avg int) int {
	if root == nil {
		return 0
	}

	left := caculateBalance(root.Left, avg)
	right := caculateBalance(root.Right, avg)
	sum += abs(left) + abs(right)

	return root.Val + left + right - avg
}

func abs(v int) int {
	if v > 0 {
		return v
	}
	return -v
}

type Node struct {
	Val   int
	Left  *Node
	Right *Node
}

func totalScore(scores string) int64 {
	scoreArray := []byte(scores)

	var total int64

	for i := 0; i < len(scoreArray); i++ {
		val := scoreArray[i]
		if val == 'X' {
			total += 10
			total += getNext2ValuesIfPossible(i+1, scoreArray)
		} else if val == '/' {
			total += 10
			total -= getValueFromByte(scoreArray[i-1])
			total += getNextOneValuesIfPossible(i+1, scoreArray)
		} else if val != ' ' {
			total += getValueFromByte(val)
		}
		fmt.Println(getValueFromByte(val), total)
	}
	return total
}

func getValueFromByte(input byte) int64 {
	return int64(input) - int64('0')
}

func getNextOneValuesIfPossible(index int, scoreArray []byte) int64 {
	for j := index; j < len(scoreArray); j++ {
		if j != ' ' {
			return getValueFromByte(scoreArray[j])
		}
	}
	return 0
}

func getNext2ValuesIfPossible(index int, scoreArray []byte) int64 {
	var sum int64
	cnt := 0
	for j := index; j < len(scoreArray); j++ {
		if j != ' ' {
			sum += getValueFromByte(scoreArray[j])
			cnt++
			if cnt == 2 {
				return sum
				break
			}
		}
	}
	return 0
}

type ListNode struct {
	Val  int
	Next *ListNode
}

func partition(head *ListNode, x int) *ListNode {
	var left, right, leftHead, rightHead *ListNode

	for cur := head; cur != nil; cur = cur.Next {
		if cur.Val < x {
			if left == nil {
				left = cur
				leftHead = left
			} else {
				left.Next = cur
				left = cur
			}
		} else {
			if right == nil {
				right = cur
				rightHead = right
			} else {
				right.Next = cur
				right = cur
			}
		}
	}
	right.Next = nil
	if leftHead != nil {
		left.Next = rightHead
		return leftHead
	}
	return rightHead
}

func summaryRanges(nums []int) []string {
	// 遍历，用两个变量
	ret := []string{}
	if len(nums) == 0 {
		return ret
	}
	if len(nums) == 1 {
		ret = append(ret, fmt.Sprintf("%d", nums[0]))
		return ret
	}
	leftIndex := 0
	for i := 1; i < len(nums); i++ {
		if nums[i]-nums[i-1] == 1 {
			continue
		}
		if leftIndex == i-1 {
			ret = append(ret, fmt.Sprintf("%d", nums[leftIndex]))
		} else {
			ret = append(ret, fmt.Sprintf("%d->%d", nums[leftIndex], nums[i-1]))
		}
		leftIndex = i
	}
	if leftIndex < len(nums)-1 {
		ret = append(ret, fmt.Sprintf("%d->%d", nums[leftIndex], nums[len(nums)-1]))
	} else {
		ret = append(ret, fmt.Sprintf("%d", nums[leftIndex]))
	}
	return ret
}

func Solution(A []int, B []int, N int) int {
	// Implement your solution here

	connectCities := make([][]int, N+1)

	cityDegree := make([]int, N+1)

	for i, _ := range A {
		c1 := A[i]
		c2 := B[i]
		connectCities[c1] = append(connectCities[c1], c2)
		connectCities[c2] = append(connectCities[c2], c1)
		cityDegree[c1]++
		cityDegree[c2]++
	}

	maxRank := 0
	for i, _ := range A {
		c1 := A[i]
		c2 := B[i]
		rank := cityDegree[c1] + cityDegree[c2] - 1
		if rank > maxRank {
			maxRank = rank
		}

	}
	return maxRank
}

func isSubsequence(s string, t string) bool {
	sa := []byte(s)
	st := []byte(t)

	currentIndex := 0
	for _, v := range sa {
		index := getIndex(st, v, currentIndex)
		if index == -1 {
			return false
		}
		currentIndex = index + 1
	}
	return true
}

func getIndex(input []byte, v byte, startIndex int) int {
	if startIndex >= len(input) {
		return -1
	}

	for i := startIndex; i < len(input); i++ {
		if input[i] == v {
			return i
		}
	}
	return -1
}

func lengthOfLastWord(s string) int {
	input := []byte(s)
	lastLen := 0
	cnt := 0
	for i := 0; i < len(input); i++ {
		if input[i] == ' ' {
			if cnt > 0 {
				lastLen = cnt
			}
			cnt = 0
		} else {
			cnt++
		}
	}
	if cnt > 0 {
		lastLen = cnt
	}
	return lastLen
}

func candy(ratings []int) int {
	// 1234  1432
	n := len(ratings)

	leftR := make([]int, n)
	rightR := make([]int, n)
	leftR[0] = 1
	for i := 1; i < n; i++ {
		if ratings[i] > ratings[i-1] {
			leftR[i] = leftR[i-1] + 1
		} else {
			leftR[i] = 1
		}
	}
	fmt.Println(leftR)

	var ret int
	rightR[n-1] = 1
	ret += getMax(leftR[n-1], rightR[n-1])
	for i := n - 2; i >= 0; i-- {
		if ratings[i] > ratings[i+1] {
			rightR[i] = rightR[i+1] + 1
		} else {
			rightR[i] = 1
		}
		ret += getMax(leftR[i], rightR[i])
	}
	fmt.Println(rightR)
	return ret
}

func getMax(left, right int) int {
	if left > right {
		return left
	}
	return right
}

func calculateCombine(input []byte, cntMap map[string]int) int {
	if len(input) <= 1 {
		if len(input) == 1 && input[0] == '0' {
			return 0
		}
		return 1
	}

	subString1 := input[0]
	subString2 := input[1]
	if subString1 == '0' {
		return 0
	}

	k1 := string(input[1:])
	if subString1 > '2' || subString1 == '0' || (subString1 == '2' && subString2 > '6') {
		if cntMap[k1] != 0 {
			return cntMap[k1]
		}
		cntMap[k1] = calculateCombine(input[1:], cntMap)
		return cntMap[k1]
	}

	var cnt1 int
	if cntMap[k1] != 0 {
		cnt1 = cntMap[k1]
	} else {
		cnt1 = calculateCombine(input[1:], cntMap)
	}

	k2 := string(input[2:])
	var cnt2 int
	if cntMap[k2] != 0 {
		cnt2 = cntMap[k2]
	} else {
		cnt2 = calculateCombine(input[2:], cntMap)
	}

	k := string(input)
	cntMap[k] = cnt1 + cnt2
	return cntMap[k]
}
